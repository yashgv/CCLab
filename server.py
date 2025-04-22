from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import sqlite3
import re
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from langchain.chains import create_sql_query_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(
    title="Healthcare Data Analysis API",
    description="API for healthcare data analysis and visualization",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    sql_query: str
    data: List[Dict[str, Any]]
    plot_image: Optional[str] = None
    message: Optional[str] = None

csv_file_path = 'Healthcare_dataset_with_summary.csv'
database_path = 'data.db'
connection = None
db = None
df = None

def initialize_database():
    global connection, db, df
    try:
        df = pd.read_csv(csv_file_path)
        connection = sqlite3.connect(database_path)
        df.to_sql('patients', connection, if_exists='replace', index=False)
        db = SQLDatabase.from_uri(f"sqlite:///{database_path}")
        logger.info("Database initialization completed successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise

def get_llm():
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            convert_system_message_to_human=True,
            temperature=0.0
        )
    except Exception as e:
        logger.error(f"LLM initialization failed: {str(e)}")
        raise

def generate_plot(df: pd.DataFrame, question: str) -> Optional[str]:
    try:
        sns.set_theme(style="whitegrid")
        plt.clf()
        numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        plt.figure(figsize=(10, 6))

        if len(numerical_columns) >= 1 and len(categorical_columns) >= 1:
            num_col = numerical_columns[0]
            cat_col = categorical_columns[0]
            agg_df = df.groupby(cat_col)[num_col].mean().reset_index()
            sns.barplot(data=agg_df, x=cat_col, y=num_col, palette="Blues_d")
        elif len(numerical_columns) >= 2:
            sns.scatterplot(data=df, x=numerical_columns[0], y=numerical_columns[1], palette="viridis")
        elif len(numerical_columns) == 1:
            sns.histplot(data=df, x=numerical_columns[0], kde=True, color='skyblue')
        elif len(categorical_columns) == 1:
            count_df = df[categorical_columns[0]].value_counts().reset_index()
            count_df.columns = [categorical_columns[0], 'counts']
            sns.barplot(data=count_df, x=categorical_columns[0], y='counts', palette="Set3")
        else:
            return None

        plt.xticks(rotation=45)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        buf_image = Image.open(buf)
        compressed_buf = io.BytesIO()
        buf_image.save(compressed_buf, format='PNG', optimize=True, quality=85)
        compressed_buf.seek(0)
        return base64.b64encode(compressed_buf.getvalue()).decode()
    except Exception as e:
        logger.error(f"Plot generation failed: {str(e)}")
        return None

@app.on_event("startup")
async def startup_event():
    global df
    df = initialize_database()

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        llm = get_llm()
        chain = create_sql_query_chain(llm, db)
        response = chain.invoke({"question": request.question})
        
        # Enhanced query cleanup
        sql_query = response
        # Remove common prefixes and text
        prefixes_to_remove = [
            'Question:.*?SQLQuery:', 'SQLQuery:', 'SQL:', 'Query:',
            'Question:.*?\n'
        ]
        for prefix in prefixes_to_remove:
            sql_query = re.sub(prefix, '', sql_query, flags=re.DOTALL)
        
        # Remove markdown code blocks
        sql_query = re.sub(r'^```sql\n|\n```$', '', sql_query)
        # Clean up whitespace
        sql_query = sql_query.strip()
        
        if not sql_query:
            raise ValueError("Failed to generate valid SQL query")
            
        df_query_result = pd.read_sql_query(sql_query, connection)

        if df_query_result.empty:
            return QueryResponse(sql_query=sql_query, data=[], message="The query returned no results.")

        plot_image = generate_plot(df_query_result, request.question)
        message = "Query processed successfully with visualization." if plot_image else "Query processed successfully, but no suitable visualization could be generated."

        return QueryResponse(
            sql_query=sql_query,
            data=df_query_result.to_dict(orient='records'),
            plot_image=plot_image,
            message=message
        )
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": str(e), "timestamp": datetime.now().isoformat()})

@app.get("/tables")
async def get_tables():
    try:
        return {"tables": db.get_usable_table_names()}
    except Exception as e:
        logger.error(f"Failed to get tables: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/schema/{table_name}")
async def get_table_schema(table_name: str):
    try:
        query = f"SELECT * FROM {table_name} LIMIT 0;"
        df_schema = pd.read_sql_query(query, connection)
        return {
            "table_name": table_name,
            "columns": [{"name": col, "type": str(df_schema[col].dtype)} for col in df_schema.columns]
        }
    except Exception as e:
        logger.error(f"Failed to get schema for table {table_name}: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Table {table_name} not found")

@app.get("/health")
async def health_check():
    try:
        return {
            "status": "healthy",
            "database_connected": connection is not None,
            "tables_available": db.get_usable_table_names() if db else [],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dataset/info")
async def get_dataset_info():
    try:
        return {
            "total_records": len(df),
            "columns": list(df.columns),
            "summary": df.describe(include='all').to_dict(),
            "missing_values": df.isnull().sum().to_dict()
        }
    except Exception as e:
        logger.error(f"Failed to get dataset info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
