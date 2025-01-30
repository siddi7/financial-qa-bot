```python
import os
import gradio as gr
from dotenv import load_dotenv
from financial_qa_bot import FinancialQABot

# Load environment variables
load_dotenv()

def create_gradio_interface():
    """Create Gradio interface for Financial QA Bot"""
    try:
        qa_bot = FinancialQABot()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize QA Bot: {str(e)}")

    with gr.Blocks(title="Financial Statement Q&A Bot") as interface:
        gr.Markdown("# Financial Statement Q&A Bot")
        gr.Markdown("""
        ### Upload your financial statement PDF and ask questions about the data
        Example questions:
        - What is the total revenue?
        - How much were the operating expenses?
        - What was the profit margin?
        - Show me the quarterly earnings
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Upload Financial Statement PDF",
                    file_types=[".pdf"]
                )
                process_button = gr.Button(
                    "Process Financial Statement",
                    variant="primary"
                )
                pdf_output = gr.Textbox(
                    label="PDF Processing Results",
                    lines=5,
                    interactive=False
                )
            
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Financial Q&A",
                    height=400
                )
                msg = gr.Textbox(
                    label="Ask a question about the financial data",
                    placeholder="e.g., What was the total revenue?"
                )
                clear = gr.Button("Clear Chat History")
            with gr.Column(scale=1):
                context_box = gr.Textbox(
                    label="Supporting Financial Data",
                    lines=10,
                    interactive=False
                )
        
        def process_pdf_handler(file):
            if file is None:
                return "Please upload a PDF file."
            summary, _ = qa_bot.process_pdf(file.name)
            return summary
        
        def respond(message, chat_history):
            if not message.strip():
                return chat_history, "", ""
            
            answer, relevant_data = qa_bot.answer_question(message)
            history = chat_history + [[message, answer]]
            return history, "", relevant_data
        
        def clear_chat():
            return [], ""
        
        # Wire up the interface
        process_button.click(
            process_pdf_handler,
            inputs=[file_input],
            outputs=[pdf_output]
        )
        
        msg.submit(
            respond,
            [msg, chatbot],
            [chatbot, msg, context_box]
        )
        
        clear.click(
            clear_chat,
            None,
            [chatbot, context_box]
        )
        
    return interface

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(debug=True, share=True)
```

%%writefile /content/drive/MyDrive/financial-qa-bot/src/financial_qa_bot.py
``` python
import os
from typing import List, Dict, Tuple
import pandas as pd
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from .utils.pdf_processor import PDFProcessor

class FinancialQABot:
    def __init__(self):
        # Initialize with environment variables
        self._initialize_apis()
        self._setup_models()
        self._setup_pinecone()
        self.pdf_processor = PDFProcessor()
        self.current_pdf_data = None

    def _initialize_apis(self):
        """Initialize API configurations"""
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        
        if not self.gemini_api_key or not self.pinecone_api_key:
            raise ValueError("Missing required API keys in environment variables")
            
        genai.configure(api_key=self.gemini_api_key)

    def _setup_models(self):
        """Initialize ML models"""
        self.llm = genai.GenerativeModel('gemini-pro')
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = len(self.encoder.encode("test"))

    def _setup_pinecone(self):
        """Initialize Pinecone index"""
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index_name = "financial-qa-index"
        
        try:
            if self.index_name in self.pc.list_indexes().names():
                self.pc.delete_index(self.index_name)
                
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            self.index = self.pc.Index(self.index_name)
        except Exception as e:
            raise RuntimeError(f"Failed to setup Pinecone index: {str(e)}")

    def process_pdf(self, file_path: str) -> Tuple[str, pd.DataFrame]:
        """Process PDF and extract financial data"""
        try:
            df = self.pdf_processor.extract_data(file_path)
            if not df.empty:
                self.current_pdf_data = df
                self._index_data(df)
                
                summary = f"Successfully processed PDF. Found {len(df)} financial data points.\n"
                summary += "\nKey financial metrics found:\n"
                summary += df.head(5).to_string()
                
                return summary, df
            return "No financial data found in the PDF.", pd.DataFrame()
        except Exception as e:
            return f"Error processing PDF: {str(e)}", pd.DataFrame()

    def _index_data(self, df: pd.DataFrame, batch_size: int = 50):
        """Index financial data in batches"""
        if df.empty:
            return
            
        try:
            self.index.delete(delete_all=True)
            
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i+batch_size]
                vectors_to_upsert = self._prepare_vectors(batch_df)
                self.index.upsert(vectors=vectors_to_upsert)
        except Exception as e:
            raise RuntimeError(f"Failed to index data: {str(e)}")

    def _prepare_vectors(self, df: pd.DataFrame) -> List[Dict]:
        """Prepare vectors for indexing"""
        vectors = []
        for idx, row in df.iterrows():
            context = f"{row['Item']}: ${row['Value']:,.2f}\nFull context: {row['Original_Text']}"
            embedding = self.encoder.encode(context)
            
            vectors.append({
                "id": str(idx),
                "values": embedding.tolist(),
                "metadata": {
                    "text": context,
                    "item": row['Item'],
                    "value": row['Value']
                }
            })
        return vectors

    def answer_question(self, question: str) -> Tuple[str, str]:
        """Answer questions about financial data"""
        try:
            if self.current_pdf_data is None:
                return "Please upload a financial statement PDF first.", ""
                
            results = self._search_relevant_data(question)
            if not results.matches:
                return "No relevant financial information found.", ""
                
            contexts, relevant_data = self._process_search_results(results)
            answer = self._generate_answer(question, contexts)
            
            return answer, "Relevant Financial Data:\n" + "\n".join(relevant_data)
        except Exception as e:
            return f"Error processing question: {str(e)}", ""

    def _search_relevant_data(self, question: str):
        """Search for relevant data using question embedding"""
        question_embedding = self.encoder.encode(question)
        return self.index.query(
            vector=question_embedding.tolist(),
            top_k=5,
            include_metadata=True
        )

    def _process_search_results(self, results) -> Tuple[List[str], List[str]]:
        """Process search results into contexts and relevant data"""
        contexts = []
        relevant_data = []
        for match in results.matches:
            if match.metadata:
                contexts.append(match.metadata["text"])
                relevant_data.append(
                    f"{match.metadata['item']}: ${match.metadata['value']:,.2f}"
                )
        return contexts, relevant_data

    def _generate_answer(self, question: str, contexts: List[str]) -> str:
        """Generate answer using LLM"""
        prompt = f"""
        Based on the following financial data:
        {' '.join(contexts)}
        
        Question: {question}
        
        Please provide a clear and specific answer using the financial data provided. 
        Format any monetary values with proper currency symbols and commas. 
        If you cannot find the exact information, please say so clearly.
        """
        
        response = self.llm.generate_content(prompt)
        return response.text
```

%%writefile /content/drive/MyDrive/financial-qa-bot/src/utils/pdf_processor.py
```python
import PyPDF2
import pandas as pd
import re
from typing import List, Dict

class PDFProcessor:
    def __init__(self):
        self.patterns = [
            r'([\w\s\-&\.]+(?:Revenue|Income|Expense|Cost|Profit|Loss|Margin|Total|Sales|Assets|Liabilities|Equity))[:\s]+[\$]?([\d,]+(?:\.\d{2})?)',
            r'([\w\s\-&\.]+(?:Q[1-4]|Quarter|Year|Month))[:\s]+[\$]?([\d,]+(?:\.\d{2})?)',
            r'([\w\s\-&\.]+)[:\s]+[\$]?\(([\d,]+(?:\.\d{2})?)\)',
            r'([\w\s\-&\.]+)[:\s]+[\$]?(-[\d,]+(?:\.\d{2})?)'
        ]

    def extract_data(self, file_path: str) -> pd.DataFrame:
        """Extract financial data from PDF"""
        text = self._read_pdf(file_path)
        data = self._extract_financial_data(text)
        return self._create_dataframe(data)

    def _read_pdf(self, file_path: str) -> str:
        """Read PDF file and extract text"""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return ' '.join(
                page.extract_text() for page in pdf_reader.pages
            )

    def _extract_financial_data(self, text: str) -> List[Dict]:
        """Extract financial data using regex patterns"""
        data = []
        for line in text.split('\n'):
            for pattern in self.patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    item = match.group(1).strip()
                    value_str = match.group(2).replace(',', '')
                    try:
                        value = float(value_str)
                        data.append({
                            'Item': item,
                            'Value': value,
                            'Original_Text': line.strip()
                        })
                    except ValueError:
                        continue
        return data

    def _create_dataframe(self, data: List[Dict]) -> pd.DataFrame:
        """Create and clean DataFrame from extracted data"""
        df = pd.DataFrame(data)
        return df.drop_duplicates(subset=['Item', 'Value']) if not df.empty else df
```
