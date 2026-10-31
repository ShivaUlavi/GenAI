from typing import TypedDict, Optional, Dict, Any, Literal
import os
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI # Using your proxy setup from Notebook 04
from langchain_experimental.tools import PythonREPLTool
from langchain_core.prompts import PromptTemplate
from langgraph.types import interrupt, Command
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
load_dotenv()

# 1. State Schema
class DataScienceState(TypedDict):
    dataset_path: str
    target_column: str 
    dataset_summary: Optional[str]
    missing_values_summary: Optional[str]
    requires_imputation: bool
    imputation_strategy: Optional[str]
    scaling_strategy: Optional[str]
    model_type: Optional[str]            
    tuning_strategy: Optional[str]       
    hyperparameters: Optional[Dict[str, Any]]
    model_metrics: Optional[str]
    saved_model_path: Optional[str]      

# 2. HYBRID LLM INITIALIZATION
# Local model for simple, deterministic generation
local_llm = ChatOllama(model="qwen2.5-coder:3b", temperature=0)

# Frontier model for complex PyTorch/CUDA generation
frontier_llm = ChatOpenAI(model="google/gemini-3.7-flash", temperature=0) 

python_repl = PythonREPLTool()

def extract_code_block(llm_response: str) -> str:
    if "```python" in llm_response:
        return llm_response.split("```python")[1].split("```")[0].strip()
    elif "```" in llm_response:
        return llm_response.split("```")[1].split("```")[0].strip()
    return llm_response.replace("Action: python_repl_ast", "").replace("Action Input:", "").strip()


import os

def eda_node(state: DataScienceState) -> DataScienceState:
    print("--- 1. EDA NODE ---")
    eda_prompt = PromptTemplate.from_template("""
        You are a senior data scientist. A dataset is located at '{filepath}'.
        Write a single block of Python code using pandas and matplotlib to do the following:
        1. Load the dataset.
        2. Print the dataframe info.
        3. Print the missing values sum for each column.
        4. Print exactly 'REQUIRES_IMPUTATION: True' if there are ANY missing values in the dataframe, otherwise print 'REQUIRES_IMPUTATION: False'.
        5. Create a directory named '../plots' if it does not exist.
        6. Loop through all numerical columns, create a new figure using plt.figure(), generate a histogram/distribution plot for each, save it as '../plots/[column_name]_eda_distribution.png', and close the figure using plt.close().
        CRITICAL INSTRUCTION: Return ONLY valid Python code inside a ```python ``` markdown block. 
        """)
    response = (eda_prompt | local_llm).invoke({"filepath": state["dataset_path"]})
    repl_output = python_repl.invoke(extract_code_block(response.content))
    return {
        "dataset_summary": repl_output,
        "missing_values_summary": repl_output,
        "requires_imputation": "REQUIRES_IMPUTATION: True" in repl_output
    }

def preprocessing_node(state: DataScienceState) -> DataScienceState:
    print("--- 2. PREPROCESSING NODE ---")
    
    input_path = state["dataset_path"]
    base_name = os.path.basename(input_path)               
    file_stem = os.path.splitext(base_name)[0]             
    dynamic_output_path = f"../data/{file_stem}_preprocessed.csv"
    
    preprocess_prompt = PromptTemplate.from_template("""
        You are a data engineer. A dataset is at '{filepath}'. Target column: '{target}'.
        Imputation: {imputation}. Scaling: {scaling}.
        
        Write Python code using pandas and scikit-learn to:
        1. Load the dataset.
        2. Separate the target column from the features.
        3. Apply Imputation: 
           - If '{imputation}' is 'KNN', import and use KNNImputer from sklearn.impute.
           - If '{imputation}' is 'median', 'mean', or 'most_frequent', import and use SimpleImputer(strategy='{imputation}').
           - If '{imputation}' is 'Drop', drop rows with missing values.
        4. Apply '{scaling}' scaling to features if not 'None'.
        5. Recombine features and target, and save to '{output_filepath}' without the index.
        
        CRITICAL INSTRUCTION: Return ONLY valid Python code inside a ```python ``` block.
        """)
    
    response = (preprocess_prompt | local_llm).invoke({
        "filepath": input_path,
        "target": state["target_column"],
        "imputation": state.get("imputation_strategy", "None"),
        "scaling": state.get("scaling_strategy", "None"),
        "output_filepath": dynamic_output_path
    })
    
    repl_output = python_repl.invoke(extract_code_block(response.content))
    print("Preprocessing REPL Output:\n", repl_output)
    
    return {"dataset_path": dynamic_output_path}


def ask_imputation_node(state: DataScienceState) -> DataScienceState:
    print("--- PAUSED: ASK IMPUTATION ---")
    if state.get("requires_imputation"):
        human_choice = interrupt({
            "question": "Select Imputation Strategy:", 
            "options": ["median", "mean", "most_frequent", "KNN", "Drop"]
        })
        return {"imputation_strategy": human_choice}
    return {"imputation_strategy": "None"}

    
def ask_scaling_node(state: DataScienceState) -> DataScienceState:
    print("--- PAUSED: ASK SCALING ---")
    human_choice = interrupt({"question": "Select Scaling Strategy:", "options": ["StandardScaler", "MinMaxScaler", "None"]})
    return {"scaling_strategy": human_choice}

def ask_model_type_node(state: DataScienceState) -> DataScienceState:
    print("--- PAUSED: ASK MODEL TYPE ---")
    human_choice = interrupt({
        "question": "Select Model Architecture:", 
        "options": [
            "Linear (Regression/Logistic)", 
            "Random Forest", 
            "Gradient Boosting", 
            "Support Vector Machine (SVM)", 
            "PyTorch Neural Network"
        ]
    })
    return {"model_type": human_choice}

def ask_tuning_node(state: DataScienceState) -> DataScienceState:
    print("--- PAUSED: ASK TUNING STRATEGY ---")
    model = state.get("model_type", "the model")
    human_choice = interrupt({
        "question": f"Select Hyperparameter Tuning Strategy for {model}:",
        "options": ["GridSearchCV", "RandomizedSearchCV", "None"]
    })
    return {"tuning_strategy": human_choice}

import os

def sklearn_training_node(state: DataScienceState) -> DataScienceState:
    print("--- 4b. SCIKIT-LEARN TRAINING NODE ---")
    
    selected_model = state.get("model_type", "Random Forest")
    tuning_strat = state.get("tuning_strategy", "None")
    
    # 1. Dynamically generate the model output path
    input_path = state["dataset_path"]
    file_stem = os.path.splitext(os.path.basename(input_path))[0]             
    model_save_path = f"../models/{file_stem}_sklearn.pkl"
    
    sklearn_prompt = PromptTemplate.from_template("""
        Write Python code to train a machine learning model.
        
        CONFIGURATION:
        - Dataset: '{filepath}'
        - Target column: '{target}'
        - Model Type: '{model_type}'
        - Save Path: '{output_model_path}'
        
        STEPS TO EXECUTE:
        1. Load data, split X and y.
        2. Initialize '{model_type}' from scikit-learn.
        3. The variable `tuning_strategy` is already defined for you. If tuning_strategy is 'GridSearchCV' or 'RandomizedSearchCV', apply it to find the best hyperparameters. Otherwise, train normally.
        4. Create a directory named '../models' if it does not exist using the os module.
        5. Save the final trained model (or the best_estimator_) to '{output_model_path}' using joblib.
        6. Print the evaluation score, the best parameters (if tuning was applied), and a confirmation that the model was saved.
        
        CRITICAL INSTRUCTION: Return ONLY valid Python code inside a ```python ``` block.
        """)
    
    response = (sklearn_prompt | frontier_llm).invoke({
        "filepath": state["dataset_path"], 
        "target": state["target_column"],
        "model_type": selected_model,
        "output_model_path": model_save_path
    })
    
    executable_code = extract_code_block(response.content)
    
    # --- THE BULLETPROOF FIX: Code Injection ---
    # We manually inject the variable at the top of the script so it ALWAYS exists
    executable_code = f"tuning_strategy = '{tuning_strat}'\n" + executable_code
    
    print("Generated Scikit-Learn Code:\n", executable_code)
    
    repl_output = python_repl.invoke(executable_code)
    print("Scikit-Learn REPL Output:\n", repl_output)
    
    return {"model_metrics": repl_output, "saved_model_path": model_save_path}

def pytorch_training_node(state: DataScienceState) -> DataScienceState:
    print("--- 4a. PYTORCH TRAINING NODE ---")
    
    tuning_strat = state.get("tuning_strategy", "None")
    
    # 1. Dynamically generate the model output path
    input_path = state["dataset_path"]
    file_stem = os.path.splitext(os.path.basename(input_path))[0]             
    model_save_path = f"../models/{file_stem}_pytorch.pt"
    
    pytorch_prompt = PromptTemplate.from_template("""
        Write Python code to train a neural network.
        
        CONFIGURATION:
        - Dataset: '{filepath}'
        - Target column: '{target}'
        - Save Path: '{output_model_path}'
        
        STEPS TO EXECUTE:
        1. Load data, split X and y, and convert to PyTorch tensors.
        2. Initialize device: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        3. Build a PyTorch nn.Module.
        4. CRITICAL DEVICE MAPPING: You MUST move the model to the device immediately after creating it using `model = model.to(device)`. You MUST also move the data to the device inside your training loop using `inputs, labels = inputs.to(device), labels.to(device)`.
        5. The variable `tuning_strategy` is already defined for you. If tuning_strategy is NOT 'None', implement a nested loop to test 2 different learning rates and 2 different hidden dimensions, and train using the best combination. Otherwise, train normally for 50 epochs using Adam lr=0.01.
        6. Create a directory named '../models' if it does not exist using the os module.
        7. Save the final trained model's state dictionary to '{output_model_path}' using torch.save().
        8. Print the final loss and a confirmation that the model was saved.
        
        CRITICAL INSTRUCTION: Return ONLY valid Python code inside a ```python ``` block.
        """)
    
    response = (pytorch_prompt | frontier_llm).invoke({
        "filepath": state["dataset_path"], 
        "target": state["target_column"],
        "output_model_path": model_save_path
    })
    
    executable_code = extract_code_block(response.content)
    
    # Code Injection to prevent NameError
    executable_code = f"tuning_strategy = '{tuning_strat}'\n" + executable_code
    
    print("Generated PyTorch Code:\n", executable_code)
    
    repl_output = python_repl.invoke(executable_code)
    print("PyTorch REPL Output:\n", repl_output)
    
    return {"model_metrics": repl_output, "saved_model_path": model_save_path}

from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

# 1. The Conditional Router Function
def route_model(state: DataScienceState) -> Literal["pytorch_training_node", "sklearn_training_node"]:
    # Route PyTorch strictly to the PyTorch node
    if state.get("model_type") == "PyTorch Neural Network":
        return "pytorch_training_node"
    
    # Route Linear, Random Forest, Gradient Boosting, and SVM to the Scikit-Learn node
    return "sklearn_training_node"

# 2. Build the Graph
builder = StateGraph(DataScienceState)

# Add all execution and HITL nodes
builder.add_node("run_eda", eda_node)
builder.add_node("ask_imputation", ask_imputation_node)
builder.add_node("ask_scaling", ask_scaling_node)
builder.add_node("apply_preprocessing", preprocessing_node)
builder.add_node("ask_model", ask_model_type_node)
builder.add_node("ask_tuning", ask_tuning_node)
builder.add_node("pytorch_training_node", pytorch_training_node)
builder.add_node("sklearn_training_node", sklearn_training_node)

# 3. Define the Sequential Flow
builder.add_edge(START, "run_eda")
builder.add_edge("run_eda", "ask_imputation")
builder.add_edge("ask_imputation", "ask_scaling")
builder.add_edge("ask_scaling", "apply_preprocessing")
builder.add_edge("apply_preprocessing", "ask_model")
builder.add_edge("ask_model", "ask_tuning")

# 4. Define the Conditional Routing (Branching happens AFTER tuning is selected)
builder.add_conditional_edges("ask_tuning", route_model)

# 5. Close the Graph
builder.add_edge("pytorch_training_node", END)
builder.add_edge("sklearn_training_node", END)

# 6. Compile with Checkpointer (Crucial for Streamlit / HITL pauses)
memory = InMemorySaver()
agent_graph = builder.compile(checkpointer=memory)

