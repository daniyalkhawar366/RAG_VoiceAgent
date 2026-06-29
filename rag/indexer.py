import os
import json
import re
import sys
import chromadb

# Reconfigure stdout for UTF-8 compatibility on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from config import DB_PATH

CARS_JSON_PATH = os.path.join(BASE_DIR, "data", "cars.json")

def build_car_description(car, year):
    year_str = f" ({year})" if year else ""
    desc = f"{car['name']}{year_str}. "
    desc += f"Price: {car['price_sar']:,} SAR. " if car['price_sar'] else "Price: Contact for pricing. "
    desc += f"Body Type: {car['body_type']}. " if car['body_type'] else ""
    desc += f"Engine/Variant: {car['engine']}. " if car['engine'] else ""
    desc += f"Transmission: {car['transmission']}. " if car['transmission'] else ""
    desc += f"Fuel Type: {car['fuel_type']}. " if car['fuel_type'] else ""
    desc += f"Exterior Color: {car['exterior_color']}. " if car['exterior_color'] else ""
    desc += f"Interior Color: {car['interior_color']}. " if car['interior_color'] else ""
    
    mileage = car.get('mileage')
    if mileage and mileage.strip() != "REGISTER YOUR INTEREST":
        desc += f"Mileage: {mileage}. "
    else:
        desc += "Mileage: 0 km (or not specified). "
        
    if car.get('features'):
        desc += f"Key Features: {', '.join(car['features'])}."
        
    return desc

def main():
    if not os.path.exists(CARS_JSON_PATH):
        print(f"Error: {CARS_JSON_PATH} not found. Please run the scraper first.")
        return
        
    print(f"Loading car listings from {CARS_JSON_PATH}...")
    with open(CARS_JSON_PATH, "r", encoding="utf-8") as f:
        cars = json.load(f)
        
    print(f"Initializing persistent ChromaDB client at: {DB_PATH}")
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Create or get collection. By default, it uses the default local SentenceTransformer embedding function (all-MiniLM-L6-v2)
    collection_name = "car_inventory"
    print(f"Creating/getting Chroma collection: '{collection_name}'...")
    collection = client.get_or_create_collection(name=collection_name)
    
    documents = []
    metadatas = []
    ids = []
    
    print("Formatting and preparing documents...")
    for idx, car in enumerate(cars):
        # Extract model year from URL
        # e.g., "https://genesis-cpo.netlify.app/en/inventory/g80-2023-2-5-prestige-rwd-long-18-uyuni-white"
        year_match = re.search(r'-(20\d{2})-', car['url'])
        year = int(year_match.group(1)) if year_match else 0
        
        # Build description for RAG semantic search
        description = build_car_description(car, year)
        documents.append(description)
        
        # Flatten metadata for Chroma (no nested lists/dicts allowed)
        metadata = {
            "url": car["url"],
            "name": car["name"],
            "price_sar": car["price_sar"] if car["price_sar"] is not None else 0,
            "year": year,
            "body_type": car["body_type"] or "",
            "exterior_color": car["exterior_color"] or "",
            "interior_color": car["interior_color"] or "",
            "fuel_type": car["fuel_type"] or "",
            "transmission": car["transmission"] or "",
            "engine": car["engine"] or "",
            "features_str": ", ".join(car["features"]) if car.get("features") else ""
        }
        metadatas.append(metadata)
        
        # Use the URL slug as the ID
        slug = car["url"].split("/")[-1]
        if not slug:
            slug = f"car_{idx}"
        ids.append(slug)
        
    print(f"Adding {len(documents)} cars to collection...")
    # Add items in chunks to be safe (though 56 is small)
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print("Indexing complete!")
    print(f"Indexed {collection.count()} total vehicles in ChromaDB.")

if __name__ == "__main__":
    main()
