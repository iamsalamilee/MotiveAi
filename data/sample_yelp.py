import random

def main():
    print("Reading full dataset...")
    with open("data/yelp_businesses.jsonl", "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    print(f"Total businesses found: {len(lines)}")
    
    # Randomly shuffle and pick exactly 4000
    sampled = random.sample(lines, 4000)
    
    with open("data/yelp_businesses_normalized.jsonl", "w", encoding="utf-8") as f:
        f.writelines(sampled)
        
    print("Successfully created yelp_businesses_normalized.jsonl with 4000 random real restaurants!")

if __name__ == "__main__":
    main()
