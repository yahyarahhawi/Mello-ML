import json

def examine_json_structure(filename):
    """Examine the structure of the JSON file"""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    print(f"Data type: {type(data)}")
    if isinstance(data, list):
        print(f"Number of entries: {len(data)}")
        print("\nFirst entry structure:")
        first_entry = data[0]
        for key, value in first_entry.items():
            print(f"  {key}: {type(value)} - {str(value)[:100]}...")
        
        print("\nSecond entry structure:")
        second_entry = data[1]
        for key, value in second_entry.items():
            print(f"  {key}: {type(value)} - {str(value)[:100]}...")
            
        print("\nKeys in first 5 entries:")
        for i in range(min(5, len(data))):
            print(f"Entry {i}: {list(data[i].keys())}")

if __name__ == "__main__":
    examine_json_structure("/Users/yahyarahhawi/Developer/Mello/Mello-ML/combined_profiles_e5_200_randomized.json") 