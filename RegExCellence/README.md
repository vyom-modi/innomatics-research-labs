# RegExCellence

RegExCellence is a web application built with Flask that allows users to test regular expressions against input strings. It provides a simple interface for entering a test string and a regex pattern, and it highlights the matched portions of the input string.

## Features

- Enter a test string and a regex pattern to match against.
- View the matched portions of the input string highlighted.
- Validate email addresses using a dedicated email validation tool.
- Simple and easy-to-use interface.

## Email Validation

RegExCellence includes a dedicated email validation tool that allows users to validate email addresses using regular expressions. Simply enter an email address and click "Validate" to check its validity.

## Usage

1. Clone this repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Flask application using `python app.py`.
4. Access the application in your web browser at `http://localhost:5000`.

## File Structure

```md
.
├── app.py
├── requirements.txt
├── static
│   └── styles.css
└── templates
    ├── index.html
    └── validate_email.html

- `app.py`: Main Flask application file.
- `requirements.txt`: List of Python dependencies.
- `static`: Directory for static files like CSS.
- `templates`: Directory for HTML templates.
