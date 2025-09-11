from utils import (
    generate_with_single_input, 
    generate_with_multiple_input, 
    generate_params_dict
)

def check_if_outfit_or_supplement(query):
    prompt = f"""
Determine the category of the following query as either "nutritional" or "outfit" related.
- Nutritional queries: These are related to nutrition products, such as whey protein, vitamins, supplements, dietary products, and health-related food and beverages.
  - Outfit queries: These pertain to clothing and fashion, including items like shirts, dresses, shoes, accessories, and jewelry.
Examples:

1. Query: “Where can I buy high-protein snacks?” Expected answer: Nutritional
2. Query: “Best shirt styles for summer 2023” Expected answer: Outfit
3. Query: “Are there any shoes designed for running?” Expected answer: Outfit
4. Query: “What multivitamins should I take daily?” Expected answer: Nutritional
5. Query: “Best weight loss products that are stylish” Expected answer: Nutritional
6. Query: “Athletic wear that boosts performance” Expected answer: Outfit 

Query: {query}

Instructions: Respond with “Nutritional” if the query pertains to nutritional products or “Outfit” if it pertains to clothing or fashion products.
Answer only one single word.
"""
    return prompt



# ASCII color codes
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

queries = [
    {"query": "Where can I buy whey protein?", "label": "Nutritional"},
    {"query": "Recommended vitamins for winter", "label": "Nutritional"},
    {"query": "Latest fashion for women's dresses", "label": "Outfit"},
    {"query": "Comfortable sneakers for daily use", "label": "Outfit"},
    {"query": "Best energy bars for athletes", "label": "Nutritional"},
    {"query": "Trendy accessories for men", "label": "Outfit"},
    {"query": "Low-carb diet food options", "label": "Nutritional"},
    {"query": "What supplements help with muscle recovery?", "label": "Nutritional"},
    {"query": "Casual wear that supports healthy living", "label": "Outfit"}
]

for item in queries:
    query = item["query"]
    prompt = check_if_outfit_or_supplement(query)
    expected_label = item["label"]
    response = generate_with_single_input(prompt, max_tokens = 2)
    result = response['content']
    
    # Determine color based on comparison
    if result == expected_label:
        color = GREEN
    else:
        color = RED

    print(f"Query: {query}\nResult: {result}\nExpected: {color}{expected_label}{RESET}\n")
    
    