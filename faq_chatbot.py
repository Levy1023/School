import os
import pandas as pd
from anthropic import Anthropic
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import json

class FitnessChatbot:
    def __init__(self, api_key=None):
        """
        Initialize the Fitness Equipment FAQ Chatbot
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = Anthropic(api_key=self.api_key)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = None
        self.documents = []
        
    def load_dataset(self, csv_path):
        """
        Load FAQ dataset from CSV file
        """
        print(f"Loading dataset from {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} records")
        
        for idx, row in df.iterrows():
            question = row['question']
            answer = row['answer']
            category = row.get('category', 'General')
            product = row.get('product', 'Unknown')
            
            content = f"Question: {question}\nAnswer: {answer}\nCategory: {category}\nProduct: {product}"
            
            doc = Document(
                page_content=content,
                metadata={
                    'question': question,
                    'answer': answer,
                    'category': category,
                    'product': product,
                    'id': idx
                }
            )
            self.documents.append(doc)
        
        return df
    
    def preprocess_and_index(self):
        """
        Preprocess documents and create vector store index
        """
        print("Preprocessing documents and creating embeddings")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        
        split_docs = text_splitter.split_documents(self.documents)
        print(f"Split into {len(split_docs)} chunks")
        
        print("Creating FAISS vector store")
        self.vector_store = FAISS.from_documents(
            split_docs,
            self.embeddings
        )
        print("Vector store created successfully")
        
    def retrieve_relevant_context(self, query, k=3):
        """
        Retrieve most relevant FAQ entries for the query
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call preprocess_and_index first.")
        
        results = self.vector_store.similarity_search(query, k=k)
        
        context = ""
        for i, doc in enumerate(results):
            context += f"FAQ {i+1}:\n{doc.page_content}\n\n"
        
        return context, results
    
    def generate_response(self, user_query, context):
        """
        Generate response using Claude API with retrieved context
        """
        system_prompt = """You are a helpful customer service assistant for a fitness equipment company. 
Your role is to answer customer questions about our products based on the FAQ information provided.

Guidelines:
- Answer questions accurately using the provided FAQ context
- Be friendly and professional
- If the answer is not in the provided context, politely say you do not have that specific information
- Keep responses concise but informative
- Format responses in a clear, easy to read manner"""

        user_message = f"""Based on the following FAQ information, please answer the customer's question.

FAQ Context:
{context}

Customer Question: {user_query}

Please provide a helpful answer based on the FAQ information above."""

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        
        return response.content[0].text
    
    def chat(self, user_query):
        """
        Main chat function that combines retrieval and generation
        """
        print(f"\nUser Query: {user_query}")
        print("Searching knowledge base...")
        
        context, retrieved_docs = self.retrieve_relevant_context(user_query)
        
        print("Generating response...")
        response = self.generate_response(user_query, context)
        
        return {
            'query': user_query,
            'response': response,
            'retrieved_docs': [doc.metadata for doc in retrieved_docs]
        }

def create_sample_dataset():
    """
    Create a sample fitness equipment FAQ dataset
    """
    data = {
        'question': [
            "What is the warranty period for the treadmill?",
            "How do I assemble the exercise bike?",
            "What is the maximum weight capacity of the elliptical machine?",
            "Do you offer free shipping on orders over $100?",
            "How do I connect my fitness tracker to the mobile app?",
            "What maintenance is required for the rowing machine?",
            "Can I return a product if I am not satisfied?",
            "What payment methods do you accept?",
            "How long does delivery typically take?",
            "Are replacement parts available for older models?",
            "What is the difference between magnetic and air resistance?",
            "How do I calibrate the heart rate monitor?",
            "What are the dimensions of the folding treadmill?",
            "Is professional installation available?",
            "How do I clean the yoga mat?",
            "What is the battery life of the wireless headphones?",
            "Can I track my workouts on multiple devices?",
            "What safety features are included with the treadmill?",
            "How do I adjust the resistance on the spin bike?",
            "Are there any subscription fees for the fitness app?",
            "What certifications do your products have?",
            "How do I update the firmware on my fitness watch?",
            "What is your price match guarantee policy?",
            "Can I cancel my order after it has been placed?",
            "How do I contact customer support?",
            "What accessories are compatible with the elliptical?",
            "Is there a mobile app for Android and iOS?",
            "How do I troubleshoot connectivity issues?",
            "What is the noise level of the treadmill motor?",
            "Do you offer student or military discounts?"
        ],
        'answer': [
            "Our treadmills come with a 5 year warranty on the frame, 2 years on parts, and 1 year on labor. Extended warranty options are available at purchase.",
            "The exercise bike comes with detailed assembly instructions. Most customers can assemble it in 45 to 60 minutes with basic tools. Assembly videos are available on our website.",
            "The elliptical machine has a maximum weight capacity of 350 pounds. This ensures safe and stable operation for most users.",
            "Yes, we offer free standard shipping on all orders over $100 within the continental United States. Expedited shipping options are available for an additional fee.",
            "Download our FitTrack app from the App Store or Google Play. Enable Bluetooth on your device and follow the pairing instructions in the app. The tracker will automatically sync your data.",
            "The rowing machine requires minimal maintenance. Wipe down the seat rail after each use, check the chain tension monthly, and inspect the footstraps regularly for wear.",
            "Yes, we offer a 30 day return policy. Products must be in original condition with all packaging and accessories. A 15% restocking fee may apply to opened items.",
            "We accept Visa, Mastercard, American Express, Discover, PayPal, Apple Pay, and Affirm financing for qualified purchases over $500.",
            "Standard shipping typically takes 5 to 7 business days. Expedited options can deliver in 2 to 3 business days. Large items may require special freight delivery which can take up to 10 business days.",
            "Yes, we maintain an inventory of replacement parts for all current models and most models from the past 10 years. Contact our parts department for availability and pricing.",
            "Magnetic resistance uses magnets to create tension and is quieter and requires less maintenance. Air resistance uses a fan and provides a more natural feel but is louder during intense workouts.",
            "To calibrate the heart rate monitor, wear the chest strap and access the settings menu. Select calibrate and follow the on screen instructions while maintaining a steady heart rate for 60 seconds.",
            "The folding treadmill measures 68 inches long by 32 inches wide when in use. When folded, it measures 68 inches long by 32 inches wide by 55 inches tall.",
            "Professional installation is available in most areas for an additional fee of $150 to $200 depending on the product. This includes assembly and placement in your desired location.",
            "Clean your yoga mat after each use with a gentle soap and water solution. Avoid harsh chemicals. Air dry completely before rolling. Deep clean monthly with a mat specific cleaner.",
            "The wireless headphones provide up to 20 hours of playback on a single charge. Charging time is approximately 2 hours using the included USB-C cable.",
            "Yes, the FitTrack app allows you to sync your workout data across multiple devices using your account login. Data is automatically backed up to the cloud.",
            "Safety features include an emergency stop clip, automatic speed reduction if clip is removed, cushioned running deck, side safety rails, and a maximum speed limit setting.",
            "The resistance knob is located below the handlebars. Turn clockwise to increase resistance and counterclockwise to decrease. Most bikes have 8 resistance levels.",
            "The basic FitTrack app is free and includes workout tracking, progress charts, and community features. Premium subscription at $9.99 per month adds guided workouts and nutrition tracking.",
            "All our products meet or exceed ASTM and EN safety standards. Select models also carry NSF certification and Energy Star ratings where applicable.",
            "Connect your fitness watch to WiFi or pair with the mobile app. Go to Settings, then System, then Software Update. Follow the prompts to download and install the latest firmware.",
            "We will match any advertised price from authorized retailers within 30 days of purchase. Contact customer service with proof of the lower price and your order number.",
            "Orders can be cancelled within 24 hours of placement at no charge. After 24 hours, cancellation may be subject to a processing fee if the order has already been prepared for shipment.",
            "Customer support is available Monday through Friday 8am to 8pm EST and Saturday 9am to 5pm EST. Call 1-800-555-0199, email support@fitnessequip.com, or use our live chat feature.",
            "Compatible accessories include tablet holders, water bottle holders, heart rate monitors, resistance bands, and workout mats. Check the product page for specific compatibility.",
            "Yes, the FitTrack app is available for both iOS and Android devices. Download from the App Store or Google Play. The app requires iOS 13.0 or Android 8.0 or higher.",
            "For connectivity issues, first ensure Bluetooth is enabled on your device. Restart both the fitness equipment and your phone. Move closer to the equipment and remove any obstacles between devices.",
            "Our treadmill motors operate at approximately 65 decibels during normal use, similar to a normal conversation. Noise levels may increase at higher speeds or inclines.",
            "Yes, we offer 10% off for students with valid ID and 15% off for active military and veterans. Discounts cannot be combined with other promotions. Verify eligibility through our partner SheerID."
        ],
        'category': [
            'Warranty', 'Assembly', 'Specifications', 'Shipping', 'Technology',
            'Maintenance', 'Returns', 'Payment', 'Shipping', 'Parts',
            'Features', 'Technology', 'Specifications', 'Installation', 'Maintenance',
            'Technology', 'Technology', 'Safety', 'Features', 'Pricing',
            'Certifications', 'Technology', 'Pricing', 'Orders', 'Support',
            'Accessories', 'Technology', 'Troubleshooting', 'Specifications', 'Pricing'
        ],
        'product': [
            'Treadmill', 'Exercise Bike', 'Elliptical', 'General', 'Fitness Tracker',
            'Rowing Machine', 'General', 'General', 'General', 'General',
            'Exercise Equipment', 'Heart Rate Monitor', 'Treadmill', 'General', 'Yoga Mat',
            'Headphones', 'Fitness Tracker', 'Treadmill', 'Spin Bike', 'FitTrack App',
            'General', 'Fitness Watch', 'General', 'General', 'General',
            'Elliptical', 'FitTrack App', 'General', 'Treadmill', 'General'
        ]
    }
    
    df = pd.DataFrame(data)
    df.to_csv('fitness_faq_dataset.csv', index=False)
    print("Sample dataset created: fitness_faq_dataset.csv")
    return df

def main():
    """
    Main function to run the chatbot
    """
    print("=== Fitness Equipment FAQ Chatbot ===\n")
    
    dataset_path = 'fitness_faq_dataset.csv'
    
    if not os.path.exists(dataset_path):
        print("Dataset not found. Creating sample dataset...")
        create_sample_dataset()
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found in environment variables")
        print("Please set your API key: export ANTHROPIC_API_KEY='your-key-here'")
        return
    
    chatbot = FitnessChatbot(api_key=api_key)
    
    df = chatbot.load_dataset(dataset_path)
    
    chatbot.preprocess_and_index()
    
    print("\n=== Chatbot Ready ===")
    print("Ask questions about our fitness equipment!")
    print("Type 'quit' to exit\n")
    
    sample_questions = [
        "What is the warranty on your treadmills?",
        "How do I clean my yoga mat?",
        "Do you offer free shipping?",
        "What is the weight limit for the elliptical?",
        "How do I connect my fitness tracker?"
    ]
    
    print("Sample questions you can ask:")
    for i, q in enumerate(sample_questions, 1):
        print(f"{i}. {q}")
    print()
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Thank you for using the Fitness Equipment FAQ Chatbot!")
            break
        
        if not user_input:
            continue
        
        try:
            result = chatbot.chat(user_input)
            print(f"\nChatbot: {result['response']}\n")
            
            print("(Retrieved from FAQs: ", end="")
            categories = [doc['category'] for doc in result['retrieved_docs']]
            print(", ".join(set(categories)) + ")\n")
            
        except Exception as e:
            print(f"Error: {str(e)}\n")

if __name__ == "__main__":
    main()
