import { useState } from 'react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [recipe, setRecipe] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Handle image selection from folder or camera
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setRecipe(null); // Reset recipe if a new photo is taken
      setError(null);
    }
  };

  // Upload to FastAPI and get the recipe
  const generateRecipe = async () => {
    if (!selectedFile) {
      setError("Please capture or select an image first.");
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:8000/upload-image', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to analyze image or generate recipe.');
      }

      const data = await response.json();
      setRecipe(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>🍳 EasyMeals AI</h1>
        <p>Snap a photo of your ingredients to get a recipe</p>
      </header>

      <main className="main-content">
        <section className="upload-section">
          <label className="camera-label">
            <input
              type="file"
              accept="image/*"
              capture="environment"
              onChange={handleFileChange}
            />
            <span className="icon">📷</span>
            {preview ? "Change Photo" : "Take Photo / Upload"}
          </label>

          {preview && (
            <div className="preview-card">
              <img src={preview} alt="Ingredient preview" className="image-preview" />
              {!recipe && (
                <button
                  onClick={generateRecipe}
                  className="generate-btn"
                  disabled={loading}
                >
                  {loading ? "Analyzing Ingredients..." : "Get My Recipe"}
                </button>
              )}
            </div>
          )}
        </section>

        {error && <div className="error-message">{error}</div>}

        {recipe && (
          <article className="recipe-result">
            <h2 className="recipe-title">{recipe.title}</h2>

            <div className="recipe-grid">
              <div className="ingredients-list">
                <h3>Ingredients</h3>
                <ul>
                  {recipe.ingredients.map((item, index) => (
                    <li key={index}>{item}</li>
                  ))}
                </ul>
              </div>

              <div className="instructions-list">
                <h3>Instructions</h3>
                <ol>
                  {recipe.directions.map((step, index) => (
                    <li key={index}>{step}</li>
                  ))}
                </ol>
              </div>
            </div>
            <button className="reset-btn" onClick={() => { setPreview(null); setRecipe(null); setSelectedFile(null); }}>
              Start Over
            </button>
          </article>
        )}
      </main>
    </div>
  );
}

export default App;