document.addEventListener("DOMContentLoaded", () => {
    fetch("http://localhost:8000/api/metrics")
        .then(res => res.json())
        .then(data => {
            document.querySelector(".average-rating").textContent = data.average_rating;
            document.querySelector(".recommendation-rate").textContent = data.recommendation_rate;
            document.querySelector(".total-reviews").textContent = data.total_reviews;
            document.querySelector(".positive-feedback").textContent = data.positive_feedback;
        });
});

// Campaign Generator
function generateCampaign() {
    const category = document.getElementById("product-category").value;
    const keywords = document.getElementById("keywords").value;
    const demographic = document.getElementById("demographic").value;

    fetch("http://localhost:8000/api/campaign", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            product_category: category,
            sentiment_keywords: keywords,
            demographic: demographic
        })
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("campaign-output").textContent = data.campaign_text;
    });
}
