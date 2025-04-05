document.addEventListener("DOMContentLoaded", function () {
    const registerForm = document.getElementById("register-form");

    if (registerForm) {
        registerForm.addEventListener("submit", function (event) {
            event.preventDefault();

            let username = document.getElementById("username").value;
            let email = document.getElementById("email").value;
            let password = document.getElementById("password").value;

            fetch("http://localhost:5000/register", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ username, email, password })
            })
            .then(response => response.json())
            .then(data => {
                const status = document.getElementById("register-status");
                if (data.message) {
                    status.textContent = data.message;
                    status.style.color = "green";
                    setTimeout(() => window.location.href = "login.html", 2000);
                } else {
                    status.textContent = data.error || "Registration failed.";
                    status.style.color = "red";
                }
            })
            .catch(error => {
                console.error("Error:", error);
                document.getElementById("register-status").textContent = "Server error. Please try again.";
            });
        });
    }
});
