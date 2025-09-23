#!/usr/bin/env python3
"""
Quick test script to debug the document loading issue
"""

import requests
import tempfile
import os
from pathlib import Path

def test_url_access():
    """Test if the URL is accessible"""
    url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    try:
        print("Testing URL accessibility...")
        response = requests.head(url, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type', 'Unknown')}")
        print(f"Content-Length: {response.headers.get('content-length', 'Unknown')}")
        
        if response.status_code == 200:
            print("✅ URL is accessible")
            return True
        else:
            print(f"❌ URL returned status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error accessing URL: {str(e)}")
        return False

def test_download_and_process():
    """Download the PDF and test local processing"""
    url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    try:
        print("Downloading PDF...")
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(response.content)
                temp_path = tmp_file.name
            
            print(f"✅ Downloaded PDF to: {temp_path}")
            print(f"File size: {len(response.content)} bytes")
            
            # Test with PyPDF2
            try:
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(temp_path)
                documents = loader.load()
                print(f"✅ Successfully loaded {len(documents)} pages with PyPDFLoader")
                
                # Show first few chars of first page
                if documents:
                    preview = documents[0].page_content[:200]
                    print(f"Preview: {preview}...")
                    
            except Exception as e:
                print(f"❌ Error with PyPDFLoader: {str(e)}")
            
            # Cleanup
            os.unlink(temp_path)
            return True
            
        else:
            print(f"❌ Download failed with status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading/processing: {str(e)}")
        return False

if __name__ == "__main__":
    print("🔍 Debugging document loading...")
    
    # Test 1: URL accessibility
    url_works = test_url_access()
    
    # Test 2: Download and local processing
    if url_works:
        download_works = test_download_and_process()
    
    print("\n📋 Summary:")
    print(f"URL Access: {'✅' if url_works else '❌'}")
    if url_works:
        print(f"Local Processing: {'✅' if download_works else '❌'}")
