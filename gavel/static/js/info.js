document.addEventListener("DOMContentLoaded", () => {
  const button = document.getElementById("info-button");
  const infoBox = document.getElementById("info-box");

  button.addEventListener("click", () => {
    const isVisible = window.getComputedStyle(infoBox).display !== "none";
    infoBox.style.display = isVisible ? "none" : "block";
  });
});