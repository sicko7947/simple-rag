"""
Command Line Interface for the RAG system
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.agent.cognitive_agent import CognitiveAgent

logger = logging.getLogger(__name__)

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

class RagCLI:
    """Command line interface for interacting with the RAG system"""
    
    def __init__(self):
        self.agent = CognitiveAgent()
        
    def upload_document(self, file_path: str) -> Dict[str, Any]:
        """
        Upload and process a document from file path
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Upload and processing results
        """
        try:
            file_path = os.path.abspath(file_path)
            
            if not os.path.exists(file_path):
                return {"success": False, "error": f"File not found: {file_path}"}
                
            # Read the file content
            with open(file_path, 'rb') as f:
                file_content = f.read()
                
            # Process the document
            result = self.agent.process_document(file_path, file_content)
            return result
            
        except Exception as e:
            logger.error(f"Error uploading document: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def ask(self, query: str) -> Dict[str, Any]:
        """
        Ask a question to the RAG system
        
        Args:
            query: User query
            
        Returns:
            Response from the agent
        """
        try:
            result = self.agent.answer_query(query)
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "query": query,
                "response": f"Error: {str(e)}",
                "success": False
            }
            
    def interactive_mode(self):
        """Start interactive session with the RAG system"""
        print("\n=== RAG System Interactive Mode ===")
        print("Type 'exit' or 'quit' to end the session")
        print("Type 'upload <file_path>' to upload a document")
        print("Type 'reset' to reset the agent's state")
        print("Type anything else to ask a question\n")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['exit', 'quit']:
                    print("Exiting interactive mode...")
                    break
                    
                elif user_input.lower() == 'reset':
                    self.agent.reset_state()
                    print("Agent state has been reset.")
                    
                elif user_input.lower().startswith('upload '):
                    file_path = user_input[7:].strip()
                    result = self.upload_document(file_path)
                    
                    if result["success"]:
                        print(f"Document uploaded and processed successfully.")
                        print(f"- Document chunks: {result.get('document_count', 'unknown')}")
                    else:
                        print(f"Error uploading document: {result.get('error', 'Unknown error')}")
                        
                else:
                    # Process as a query
                    result = self.ask(user_input)
                    
                    if result["success"]:
                        print("\nAssistant:", result["response"])
                        
                        # Print sources if available
                        sources = result.get("sources", [])
                        if sources:
                            print("\nSources:")
                            for i, source in enumerate(sources[:3]):  # Show top 3 sources
                                print(f"- {source.get('file_name', 'Unknown')}")
                    else:
                        print("\nAssistant:", result["response"])
                        
            except KeyboardInterrupt:
                print("\nExiting interactive mode...")
                break
                
            except Exception as e:
                print(f"\nError: {str(e)}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="RAG System CLI")
    parser.add_argument("--upload", "-u", help="Upload and process a document")
    parser.add_argument("--query", "-q", help="Ask a question")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive mode")
    
    return parser.parse_args()

def run_cli():
    """Main entry point for the CLI"""
    args = parse_arguments()
    cli = RagCLI()
    
    if args.upload:
        result = cli.upload_document(args.upload)
        if result["success"]:
            print(f"Document uploaded and processed successfully.")
        else:
            print(f"Error uploading document: {result.get('error', 'Unknown error')}")
            
    elif args.query:
        result = cli.ask(args.query)
        print(result["response"])
        
    elif args.interactive:
        cli.interactive_mode()
        
    else:
        print("No action specified. Use -h for help.")

if __name__ == "__main__":
    run_cli()