document.addEventListener("DOMContentLoaded", function() {
    // Hide the welcome screen only if it exists
    const welcomeScreen = document.getElementById("welcome-screen");
    if (welcomeScreen) {
        setTimeout(() => {
            welcomeScreen.classList.add("hidden");
        }, 1000);
    }

    // Smooth scrolling for navigation
    document.querySelectorAll('nav ul li a').forEach(anchor => {
        anchor.addEventListener('click', function(event) {
            event.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
});
