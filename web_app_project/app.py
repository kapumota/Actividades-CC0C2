from flask import Flask, request, render_template_string

app = Flask(__name__)

# Plantilla HTML simple
html_template = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Web App</title>
</head>
<body>
    <h1>Enter something and see it echoed back!</h1>
    <form method="post" action="/">
        <input type="text" name="user_input" required>
        <button type="submit">Submit</button>
    </form>
    {% if message %}
        <p>{{ message }}</p>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    message = ""
    if request.method == 'POST':
        user_input = request.form['user_input']
        message = f"You entered: {user_input}"
    return render_template_string(html_template, message=message)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

