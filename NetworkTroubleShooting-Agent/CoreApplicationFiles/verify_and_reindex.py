"""
Data Verification and Reindexing Script
----------------------------------------
Run this script to verify your data is loading correctly and reindex if needed.
Usage: python verify_and_reindex.py
"""

import os
import pandas as pd

def verify_data_files():
    """Verify all data files exist and show their structure."""
    print("="*70)
    print("DATA FILE VERIFICATION")
    print("="*70)
    
    data_files = [
        "data/cleaned/cleaned_src_tech_records.csv",
        "data/cleaned/cleaned_src_incident_records.csv",
        "data/cleaned/cleaned_metadata_tech_records.csv",
        "data/cleaned/cleaned_metadata_incident_records.csv",
    ]
    
    for filepath in data_files:
        print(f"\n📄 {filepath}")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            print(f"   ✅ Found: {len(df)} records")
            print(f"   📋 Columns: {', '.join(df.columns.tolist())}")
            
            # Check for key columns
            has_problem = any('problem' in c.lower() or 'description' in c.lower() for c in df.columns)
            has_solution = any('solution' in c.lower() or 'steps' in c.lower() or 'detail' in c.lower() for c in df.columns)
            
            print(f"   🔍 Has Problem field: {has_problem}")
            print(f"   🔧 Has Solution field: {has_solution}")
            
            # Show sample data
            if not df.empty:
                print(f"   📊 Sample record:")
                first_row = df.iloc[0]
                for col in df.columns[:5]:  # Show first 5 columns
                    value = str(first_row[col])[:50]
                    print(f"      {col}: {value}...")
        else:
            print(f"   ❌ NOT FOUND")
    
    print("\n" + "="*70)

def test_retriever_loading():
    """Test if the retriever loads data correctly."""
    print("\nTESTING RETRIEVER DATA LOADING")
    print("="*70)
    
    try:
        from step2_retriever_qdrant import load_data
        df = load_data()
        
        print(f"\n✅ Successfully loaded {len(df)} records")
        print(f"\n📊 Data Summary:")
        print(f"   - Problem texts (non-empty): {(df['problem_text'].fillna('') != '').sum()}")
        print(f"   - Solution texts (non-empty): {(df['solution_text'].fillna('') != '').sum()}")
        print(f"   - Both problem & solution: {((df['problem_text'].fillna('') != '') & (df['solution_text'].fillna('') != '')).sum()}")
        
        print(f"\n📋 Source Distribution:")
        print(df['source'].value_counts().to_string())
        
        print(f"\n🔍 Sample Records:")
        for idx in range(min(3, len(df))):
            row = df.iloc[idx]
            print(f"\n   Record {idx + 1}:")
            print(f"   Source: {row['source']}")
            print(f"   Problem: {row['problem_text'][:80]}...")
            print(f"   Solution: {row['solution_text'][:80]}...")
            print(f"   Product: {row['product_id']} | Doc: {row['doc_id']}")
        
        # Check for empty records
        empty_records = df[(df['problem_text'].fillna('') == '') & (df['solution_text'].fillna('') == '')]
        if len(empty_records) > 0:
            print(f"\n⚠️ WARNING: Found {len(empty_records)} records with BOTH problem and solution empty!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False

def reindex_data():
    """Reindex the data in Qdrant."""
    print("\n\nREINDEXING DATA")
    print("="*70)
    
    try:
        from step2_retriever_qdrant import setup_qdrant_collection, index_data
        
        print("Setting up Qdrant collection...")
        setup_qdrant_collection()
        
        print("\nIndexing data...")
        index_data()
        
        print("\n✅ Reindexing complete!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during reindexing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sample_query():
    """Test a sample query to verify retrieval works."""
    print("\n\nTESTING SAMPLE QUERY")
    print("="*70)
    
    try:
        from step2_retriever_qdrant import retrieve_qdrant
        
        test_queries = [
            "BGP routing issues",
            "IP address conflict",
            "The network isn't working"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            results = retrieve_qdrant(query, limit=3)
            
            for r in results[:3]:
                print(f"\n   Rank {r['rank']} | Score: {r['score']:.3f} | Source: {r['source']}")
                print(f"   Problem: {r['problem'][:60]}...")
                print(f"   Solution: {r['solution'][:60]}...")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during query test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n")
    print("🔧 BlueCom Data Verification & Reindexing Tool")
    print("="*70)
    
    # Step 1: Verify files
    verify_data_files()
    
    # Step 2: Test loading
    if not test_retriever_loading():
        print("\n⚠️ Data loading failed. Please check your CSV files.")
        exit(1)
    
    # Step 3: Ask user if they want to reindex
    print("\n" + "="*70)
    response = input("\nDo you want to reindex the data? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        if reindex_data():
            # Step 4: Test retrieval
            test_sample_query()
    else:
        print("\n✅ Skipping reindexing.")
        # Still test retrieval
        test_sample_query()
    
    print("\n" + "="*70)
    print("✅ Verification complete!")
    print("="*70)