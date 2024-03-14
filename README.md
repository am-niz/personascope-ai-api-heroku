```markdown
# PersonaScope AI

PersonaScope AI is a FastAPI application that leverages machine learning models to predict gender, age, and emotion from images. It aims to provide a comprehensive analysis of faces in images, offering insights into demographic and emotional states.

## Features

- **Gender Prediction:** Determines the gender of individuals in the image.
- **Age Prediction:** Estimates the age range of individuals in the image.
- **Emotion Prediction:** Identifies the predominant emotion from facial expressions in the image.

## Installation

Follow these steps to set up the PersonaScope AI environment:

1. Clone the PersonaScope AI repository:

```bash
git clone https://github.com/am-niz/PersonaScopeAI.git
cd PersonaScopeAI
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Start the PersonaScope AI application with the following command:

```bash
uvicorn main:app --host 0.0.0.0 --port 10000 --reload
```

After starting the server, you can visit `http://0.0.0.0:10000/docs` in your web browser to access the Swagger UI, where you can interact with the API.

## API Endpoints

- `POST /predict`: This endpoint allows you to upload an image for gender, age, and emotion prediction.

## Models

The application uses three machine learning models:
- **Gender Prediction Model**: A model trained to identify gender from facial features.
- **Age Prediction Model**: Estimates the age based on facial characteristics.
- **Emotion Prediction Model**: Recognizes facial expressions to determine emotions.

## Requirements

For a detailed list of required packages, refer to the `requirements.txt` file.

## License

PersonaScope AI is open-source software licensed under the MIT License - see the LICENSE file for more details.

## Acknowledgments

- Gratitude to the developers and community around FastAPI, TensorFlow, OpenCV, and Pillow for their invaluable tools and resources.
- Special thanks to the providers of the pre-trained models used in this project for their contributions to the open-source community.

```
