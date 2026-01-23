# EasyMeals 🍽️

**Transform your ingredients into delicious meals with AI-powered recipe generation**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org/)

---

## 🎯 Inspiration

Meal planning is surprisingly difficult for college students—limited time, limited ingredients, and frequent food waste make cooking feel harder than it should be. Most students know what's in their fridge, but not what to cook with it. 

**EasyMeals** was inspired by a simple question: *What if you could take a photo of your ingredients and instantly get meal ideas?*

---

## 💡 What it does

**EasyMeals** is a computer vision-powered meal recommendation app that helps users plan meals using the ingredients they already have. 

- 📸 **Upload an image** of your available ingredients
- 🤖 **AI detects** food items using YOLO object detection
- 🍳 **Generate personalized recipes** using T5 transformer models
- 🎨 **Clean, beautiful interface** for exploring meal ideas

Instead of manually searching for recipes, users receive fast, relevant meal ideas tailored to what's actually available.

---

## 🏗️ How we built it

EasyMeals is designed as a modular, full-stack AI system:

### **Backend Architecture**
- **Computer Vision**: YOLOv8 for real-time ingredient detection
- **Recipe Generation**: T5 transformer model (`flax-community/t5-recipe-generation`)
- **Recipe Parsing**: GPT-4 via LangChain for structured recipe formatting
- **API Framework**: FastAPI with asynchronous endpoints
- **Image Processing**: OpenCV for preprocessing and enhancement

### **Frontend**
- **Framework**: React for responsive UI
- **Styling**: Modern CSS with interactive components
- **API Integration**: Fetch API for seamless backend communication

### **Key Features**
- Image upload with drag-and-drop support
- Real-time ingredient detection visualization
- Structured recipe output (title, ingredients, directions)
- Error handling and loading states

Each component is loosely coupled, allowing the system to be extended and iterated on independently.

---

## 🚀 Setup

### Prerequisites

- **Python** 3.8 or higher
- **Node.js** 14 or higher
- **pip** and **npm** package managers

### Installation

1. **Clone the repository**
```bash
   git clone https://github.com/sohamgarg2020/easymeals.git
   cd easymeals
```

2. **Set up the backend**
```bash
   cd backend
   pip install -r requirements.txt
```

3. **Configure environment variables**
```bash
   # Create .env file in backend directory
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

4. **Start the backend server**
```bash
   uvicorn main:app --reload
```
   Backend will run at `http://localhost:8000`

5. **Set up the frontend** (in a new terminal)
```bash
   cd ../frontend
   npm install
```

6. **Start the frontend**
```bash
   npm start
```
   Frontend will run at `http://localhost:3000`

---

## 📦 Project Structure
```
easymeals/
├── backend/
│   ├── main.py                 # FastAPI server and endpoints
│   ├── detect.py               # YOLO ingredient detection
│   ├── generate_recipes.py    # T5 recipe generation
│   ├── agent.py                # GPT-4 recipe parsing
│   ├── requirements.txt        # Python dependencies
│   └── .env                    # Environment variables
├── frontend/
│   ├── src/
│   │   ├── App.js             # Main React component
│   │   └── ...
│   ├── package.json           # Node dependencies
│   └── public/
└── README.md
```

---

## 🎮 Usage

1. **Open the app** at `http://localhost:3000`
2. **Upload an image** of your ingredients (fruits, vegetables, pantry items)
3. **Wait for detection** - YOLO will identify food items
4. **Get your recipe** - AI generates a complete recipe with:
   - Recipe title
   - Ingredient list with measurements
   - Step-by-step cooking directions
   - Detected items summary

---

## 🛠️ Technologies Used

| Component | Technology |
|-----------|-----------|
| **Object Detection** | YOLOv8 (Ultralytics) |
| **Recipe Generation** | T5 Transformer Model |
| **Recipe Parsing** | GPT-4 (OpenAI) via LangChain |
| **Backend Framework** | FastAPI |
| **Frontend** | React |
| **Image Processing** | OpenCV |
| **Deep Learning** | PyTorch, Transformers (HuggingFace) |

---

## 🔮 What's next for EasyMeals

- 🎯 **Improved Detection**: Train on larger food datasets for better accuracy
- 📊 **Nutritional Tracking**: Add calorie counts and macro breakdowns
- 🥗 **Dietary Preferences**: Support vegan, gluten-free, allergen-free options
- 🛒 **Smart Shopping**: Suggest missing ingredients and grocery alternatives
- 📱 **Mobile App**: Deploy as a mobile-first progressive web app
- 👥 **Social Features**: Share recipes and meal plans with friends
- 🌍 **Multi-language**: Support international cuisines and languages


---

## 🙏 Acknowledgments

- [YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [HuggingFace](https://huggingface.co/) for the T5 recipe generation model
- [OpenAI](https://openai.com/) for GPT-4 API
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent API framework

---

**⭐ If you found this project helpful, please give it a star!**
