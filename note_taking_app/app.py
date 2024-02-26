from flask import Flask, render_template, request, redirect, url_for, flash
from math import ceil
from jinja2 import Environment

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for session management

# Sample notes data
# notes = [f"Note {i+1}" for i in range(50)]  # Sample notes for demonstration
notes = []

@app.route('/', methods=["GET", "POST"])
def index():
    page = request.args.get('page', 1, type=int)
    per_page = 10  # Number of notes per page
    start = (page - 1) * per_page
    end = start + per_page

    if request.method == "POST":
        if 'delete' in request.form:
            note_index = int(request.form['delete'])
            del notes[note_index]
            flash("Note deleted successfully.")
            # Recalculate pagination parameters after deleting a note
            num_notes = len(notes)
            num_pages = ceil(num_notes / per_page)
            if page > num_pages:
                page = num_pages  # Adjust page number if it exceeds the new number of pages
            start = (page - 1) * per_page
            end = start + per_page
        elif 'note' in request.form:
            note = request.form.get("note")
            if note and note.strip():  # Check if note is not empty or contains only whitespace
                notes.append(note)
                # Recalculate pagination parameters after adding a new note
                num_notes = len(notes)
                num_pages = ceil(num_notes / per_page)
                page = num_pages  # Navigate to the last page after adding a new note
                start = (page - 1) * per_page
                end = start + per_page
            else:
                flash("Note cannot be empty.")

    paginated_notes = notes[start:end]
    num_notes = len(notes)  # Get the total number of notes
    num_pages = ceil(num_notes / per_page)  # Calculate total number of pages
    return render_template("home.html", notes=paginated_notes, page=page, num_notes=num_notes, per_page=per_page, num_pages=num_pages)

if __name__ == '__main__':
    app.run(debug=True)