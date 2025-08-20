const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;
document.getElementById("uploadForm").onsubmit = async function(e) {
    e.preventDefault();
    let formData = new FormData(this);
    
    let response = await fetch("/predict/", {
        method: "POST",
        body: formData,
        headers: {
            "X-CSRFToken": csrfToken
        }
    });
    
    let result = await response.json();
    document.getElementById("results").style.display = "block";
    document.getElementById("prediction-text").textContent = result.prediction;
    document.getElementById("confidence-text").textContent = result.confidence + "%";
    document.getElementById("output-video").src = result.video_url;
}

document.addEventListener("DOMContentLoaded", () => {
    const select = document.getElementById("modelSelect");
    const selected = select.querySelector(".selected");
    const options = select.querySelector(".options");
    const hiddenInput = document.getElementById("modelInput");

    // Toggle dropdown
    selected.addEventListener("click", () => {
        options.style.display = options.style.display === "block" ? "none" : "block";
    });

    // Pick option
    options.querySelectorAll("div").forEach(option => {
        option.addEventListener("click", () => {
            selected.textContent = option.textContent;
            hiddenInput.value = option.dataset.value;
            options.style.display = "none";
        });
    });

    // Close dropdown when clicking outside
    document.addEventListener("click", (e) => {
        if (!select.contains(e.target)) {
            options.style.display = "none";
        }
    });
});
