import json
import random
import uuid
import os
from datetime import datetime, timedelta

def generate_jumia_reviews(n=30000):
    # Grouping by tier helps the AI properly map "Price Sensitivity" later
    catalog = {
        "Phones & Tablets": {
            "Premium": ["iPhone 15 Pro Max", "Samsung Galaxy S24 Ultra", "iPad Pro 12.9", "Samsung Z Fold 5"],
            "Mid": ["Samsung Galaxy A54", "Redmi Note 13 Pro", "Tecno Phantom V", "Oppo Reno 10"],
            "Budget": ["Infinix Hot 40", "Tecno Spark 20", "Itel A70", "Nokia C22", "Oraimo 20000mAh Powerbank"]
        },
        "Computing": {
            "Premium": ["MacBook Pro M3", "Dell XPS 15", "Alienware Aurora", "HP Spectre x360"],
            "Mid": ["Lenovo ThinkPad E14", "HP Pavilion 15", "Acer Swift 3", "Asus VivoBook"],
            "Budget": ["HP Stream 11", "Lenovo IdeaPad 1", "Wireless Mouse", "1TB External HDD", "Laptop Stand"]
        },
        "Home & Appliances": {
            "Premium": ["LG 86-inch 4K TV", "Samsung Double Door Fridge", "Dyson Vacuum", "Hisense Laser TV"],
            "Mid": ["Hisense 50-inch Smart TV", "Haier Thermocool Chest Freezer", "LG Front Load Washing Machine", "Century Gas Cooker"],
            "Budget": ["Binatone Blender", "Qasa Standing Fan", "Electric Kettle", "Pressing Iron", "Extension Board"]
        },
        "Fashion": {
            "Premium": ["Original Rolex Watch", "Nike Air Jordan 1", "Gucci Leather Belt", "Designer Agbada Material"],
            "Mid": ["Nike Air Force 1", "Adidas Yeezy Slides", "Corporate Two-Piece Suit", "Quality Chinos Trousers"],
            "Budget": ["Plain White Tee", "Ankara Print Fabric", "Rubber Slippers", "Faux Leather Wallet", "Baseball Cap"]
        },
        "Health & Beauty": {
            "Premium": ["Dior Sauvage Perfume", "La Mer Face Cream", "Philips OneBlade Pro", "Tom Ford Oud Wood"],
            "Mid": ["CeraVe Cleanser", "Vitamin C Serum", "Gillette Mach 3", "MAC Foundation"],
            "Budget": ["Nivea Body Lotion", "Axe Body Spray", "Dettol Soap Pack", "Toothpaste", "Hair Clipper"]
        },
        "Groceries": {
            "Premium": ["Moet & Chandon Champagne", "Glenfiddich 18yo", "Imported Olive Oil", "Basmati Rice 20kg"],
            "Mid": ["Golden Penny Pasta Carton", "Corned Beef Pack", "Hollandia Yoghurt Carton", "Nescafe Gold"],
            "Budget": ["Indomie Noodles Carton", "Gino Tomato Paste", "Dangote Sugar 1kg", "Peak Milk Sachet Pack"]
        },
        "Gaming": {
            "Premium": ["PlayStation 5 Console", "Xbox Series X", "Meta Quest 3 VR", "Asus ROG Gaming Monitor"],
            "Mid": ["Nintendo Switch OLED", "PS5 DualSense Controller", "JBL Gaming Headset", "Mechanical Keyboard"],
            "Budget": ["PS4 Controller", "Mouse Pad", "Thumb Grips", "HDMI 2.1 Cable"]
        }
    }

    # Contextual reviews based on the tier
    review_templates = {
        "Premium": {
            5: ["Worth the high price. Premium quality {item}.", "Omo, money is good! This {item} is too clean.", "Authentic and original. Very satisfied."],
            4: ["Great {item}, but the delivery fee was too high.", "I love it, but taking one star off because packaging was dented."],
            2: ["For the amount of money I paid, this {item} is underperforming.", "I suspect this is fake. Returning it immediately."],
            1: ["Scam! Paid millions for {item} and received a brick.", "Do not buy! Seller is selling fake premium goods."]
        },
        "Budget": {
            5: ["Best value for money. This {item} works perfectly.", "E cheap but e get level. Nice one.", "Saves me so much money, does the job!"],
            4: ["Manageable for the price. It's an okay {item}.", "Not bad, you get what you pay for."],
            2: ["E too cheap, I should have known it would break easily.", "Very flimsy {item}."],
            1: ["Useless {item}. Spoilt on the first day. Complete waste of my 2k."]
        },
        "Mid": {
            5: ["Solid {item}. Balances price and quality perfectly.", "No cap, this is exactly what I needed."],
            4: ["Good quality {item}. Will recommend.", "E try small. Not bad at all."],
            2: ["Expected better quality from this {item}.", "Just average, kinda disappointed."],
            1: ["Terrible. Stopped working after 2 days."]
        }
    }

    records = []
    start_date = datetime(2023, 1, 1)
    categories_list = list(catalog.keys())
    
    for i in range(n):
        cat = random.choice(categories_list)
        
        # 60% budget users, 30% mid, 10% premium users (realistic African e-commerce split)
        tier = random.choices(["Budget", "Mid", "Premium"], weights=[0.60, 0.30, 0.10])[0]
        item = random.choice(catalog[cat][tier])
        
        # Ratings distribution
        rating = random.choices([1, 2, 3, 4, 5], weights=[0.10, 0.05, 0.10, 0.35, 0.40])[0]
        
        # If rating is 3, fallback to a generic mid review, else use the tier-specific review
        if rating == 3:
            text = f"The {item} is just average. Nothing special."
        else:
            text = random.choice(review_templates[tier][rating]).format(item=item)
        
        random_days = random.randint(0, 365)
        review_date = start_date + timedelta(days=random_days)
        
        record = {
            "item_id": f"JUMIA_{uuid.uuid4().hex[:10]}",
            "item_title": item,
            "item_category": cat,
            "rating": rating,
            "review_text": text,
            "timestamp": review_date.isoformat() + "Z"
        }
        records.append(record)
        
    return records

if __name__ == "__main__":
    output_path = "data/raw/jumia_simulated_reviews.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Generating {30000} simulated Jumia reviews with proper tier mapping...")
    items = generate_jumia_reviews(30000)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
            
    print(f"Successfully saved 30,000 records to {output_path}")
