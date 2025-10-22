document.addEventListener("DOMContentLoaded", () => {
  const button = document.getElementById("info-button");
  const modal = document.getElementById("info-modal");
  const closeButton = document.getElementById("modal-close");

  if (!button || !modal) return;

  // Open modal
  button.addEventListener("click", () => {
    modal.classList.add("active");
    document.body.style.overflow = "hidden"; // Prevent background scroll
  });

  // Close modal function
  const closeModal = () => {
    modal.classList.remove("active");
    document.body.style.overflow = ""; // Restore scroll
  };

  // Close on close button
  if (closeButton) {
    closeButton.addEventListener("click", closeModal);
  }

  // Close on overlay click
  modal.addEventListener("click", (e) => {
    if (e.target === modal) {
      closeModal();
    }
  });

  // Close on Escape key
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && modal.classList.contains("active")) {
      closeModal();
    }
  });
});