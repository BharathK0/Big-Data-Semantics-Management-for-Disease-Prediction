from flask import Flask, request, render_template
import os
import subprocess

app = Flask(__name__)

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('result.html', message='No file provided', output='Please upload a file.')
    
    file = request.files['file']
    if file.filename == '':
        return render_template('result.html', message='No file selected', output='Please select a file to upload.')

    os.makedirs('uploads', exist_ok=True)
    image_path = os.path.join('uploads', file.filename)
    file.save(image_path)

    try:
        # Ensuring the subprocess call captures the standard output and standard error.
        output = subprocess.check_output(['python', 'files/master1.py', image_path], text=True, stderr=subprocess.STDOUT)
        return render_template('result.html', message='File uploaded successfully', output=output)
    except subprocess.CalledProcessError as e:
        return render_template('result.html', message='Error processing file', output=e.output)  # Make sure to capture output from the error.

if __name__ == '__main__':
    app.run(debug=True)
