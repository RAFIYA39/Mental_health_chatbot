document.addEventListener("DOMContentLoaded", function () {
    const loginForm = document.getElementById("login-form");

    if (loginForm) {
        loginForm.addEventListener("submit", function (event) {
            event.preventDefault();

            let username = document.getElementById("username").value;
            let password = document.getElementById("password").value;

            fetch("http://localhost:5000/login", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ username, password })
            })
            .then(response => response.json())
            .then(data => {
                const status = document.getElementById("login-status");

                if (data.success) {
                    localStorage.setItem("username", username); 
                    status.textContent = "Login successful!";
                    status.style.color = "green";
                    setTimeout(() => window.location.href = "/homepage", 1000);
                } else {
                    status.textContent = data.message || "Invalid username or password.";
                    status.style.color = "red";
                }
            })
            .catch(error => {
                console.error("Error:", error);
                document.getElementById("login-status").textContent = "Server error. Please try again.";
            });
        });
    }
});
