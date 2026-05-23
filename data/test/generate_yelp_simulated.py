import json
import random
import uuid
import os

def generate_restaurants(n=2000):
    categories = [
        "Nigerian", "Italian", "Chinese", "Fast Food", "Continental", 
        "Fusion", "Lebanese", "Indian", "New American", "Ethiopian", "Korean BBQ","Buka"
    ]
    cities = ["Lagos", "Abuja", "Port Harcourt", "Ibadan", "Kano", "Enugu", "New York", "London", "ilorin"]
    vibes = ["casual", "romantic", "trendy", "quiet", "loud", "family-friendly","local"]
    noise_levels = ["quiet", "average", "loud", "very loud"]
    price_ranges = ["₦", "₦₦", "₦₦₦", "₦₦₦₦", ]
    
    adjectives = ["Tasty", "Spicy", "Royal", "Golden", "Mama's", "Urban", "Classic", "Premium", "Chop", "Sabi"]
    nouns = ["Pot", "Kitchen", "Lounge", "Grill", "Bistro", "Diner", "Place", "Spot", "Express", "Eatery"]
    
    records = []
    for i in range(n):
        cat = random.choice(categories)
        city = random.choice(cities)
        # Create a realistic sounding restaurant name
        title = f"{random.choice(adjectives)} {cat} {random.choice(nouns)}"
        
        # 1. Enforce realistic pricing based on Nigerian market realities
        if cat in ["Lebanese", "Korean BBQ", "New American"]:
            # Foreign/Niche is almost always expensive in Nigeria
            price = random.choices(["₦₦", "₦₦₦", "₦₦₦₦"], weights=[0.1, 0.5, 0.4])[0]
        elif cat in ["Buka", "Fast Food"]:
            # Local/Fast food is usually budget-friendly
            price = random.choices(["₦", "₦₦"], weights=[0.8, 0.2])[0]
        elif cat in ["Continental", "Fusion", "Italian"]:
            # Mid-to-high
            price = random.choices(["₦₦", "₦₦₦", "₦₦₦₦"], weights=[0.3, 0.5, 0.2])[0]
        else:
            # Chinese, Nigerian, Indian vary widely
            price = random.choices(["₦", "₦₦", "₦₦₦", "₦₦₦₦"], weights=[0.2, 0.4, 0.3, 0.1])[0]

        # 2. Enforce realistic vibes based on Category & Price
        if price in ["₦₦₦", "₦₦₦₦"]:
            vibe = random.choices(["romantic", "trendy", "quiet"], weights=[0.4, 0.4, 0.2])[0]
        elif cat == "Buka":
            vibe = random.choices(["local", "casual", "loud"], weights=[0.6, 0.3, 0.1])[0]
        else:
            vibe = random.choice(vibes)

        record = {
            "item_id": f"YELP_{uuid.uuid4().hex[:10]}",
            "item_title": title,
            "categories": f"Restaurants, {cat}",
            "city": city,
            "rating": round(random.uniform(2.5, 5.0), 1),
            "attributes": {
                "ambience": random.choice(vibes),
                "noise_level": random.choice(noise_levels),
                "price_range": random.choice(price_ranges),
                "good_for_groups": str(random.choice([True, False]))
            }
        }
        records.append(record)
    return records

if __name__ == "__main__":
    output_path = "data/raw/yelp_simulated_reviews.jsonl"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Generating 2000 simulated Yelp businesses...")
    items = generate_restaurants(2000)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
            
    print(f"Successfully saved 2000 records to {output_path}")
    print("Next step: Run `python data/build_chroma_index.py --input data/raw/yelp_simulated_reviews.jsonl`")
