# Note Taking Application

This is a simple note-taking application developed as part of a project. The application allows users to create, view, complete, and delete notes.

## Features

- **Input Validation:** Ensures that users cannot submit empty notes, enhancing data integrity.
- **Security Measures:** Implements security measures such as escaping user input to prevent Cross-Site Scripting (XSS) attacks.
- **Post-Redirect-GET Pattern:** Applies the POST-Redirect-GET pattern to prevent form resubmission upon page reload.
- **Improved UI:** Enhances the user interface by center-aligning the note input field and "Add Note" button.
- **Pagination:** Introduces pagination functionality to manage and display a large number of notes effectively.
- **Complete Button:** Implements a "Complete" button feature for each note, enabling users to mark tasks as completed.
- **Delete Button:** Implements a delete button for each note, allowing users to easily remove unwanted entries. Additionally, a confirmation dialog was added to ensure that users confirm their intention before deleting a note, preventing accidental data loss.
- **Flash Messages:** Displays flash messages to provide feedback to users about the success or failure of their actions.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/vyom-modi/innomatics-research-labs.git
2. Navigate to the note_taking_app directory:

   ```bash
   cd innomatics-research-labs/note_taking_app
3. Run the application:
   ```bash
   python3 app.py
Access the application in your web browser at http://localhost:5000.
