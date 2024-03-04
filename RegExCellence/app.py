# app.py
import re
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Retrieve form data
        test_string = request.form.get("test_string", "")
        regex_pattern = request.form.get("regex", "")

        # Perform regex search
        matches = list(re.finditer(regex_pattern, test_string))

        # Convert matches to strings
        matches = [match.group() for match in matches]

        # Render template with results
        return render_template("index.html", test_string=test_string, regex=regex_pattern, matches=matches)
    else:
        # Render the form template
        return render_template("index.html")

# Define route for email validation form
@app.route("/validate_email", methods=["GET", "POST"])
def validate_email():
    if request.method == "POST":
        # Handle form submission
        email = request.form.get("email")
        # Perform email validation using regular expression
        if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            # Email is valid
            validation_result = "Valid"
        else:
            # Email is invalid
            validation_result = "Invalid"
        return render_template("validate_email.html", validation_result=validation_result, email=email)
    else:
        # Render the email validation form template
        return render_template("validate_email.html")

if __name__ == "__main__":
    app.run(debug=True)
