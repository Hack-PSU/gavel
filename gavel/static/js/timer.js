const button = document.getElementById('startButton');
const timerDisplay = document.getElementById('timer');
const timerContainer = document.getElementById('timer-container');

let ping;
let intervalId;
let totalSeconds = 0;

window.addEventListener('DOMContentLoaded', () => {
  ping = new Audio(pingSoundPath); // defined in HTML
  // Preload the audio for better playback
  ping.load();

  // Enable audio on first user interaction (required by some browsers)
  const enableAudio = () => {
    ping.play().then(() => {
      ping.pause();
      ping.currentTime = 0;
    }).catch(() => {
      // Audio not yet enabled, will try again
    });
  };

  // Try to enable audio on various user interactions
  ['click', 'touchstart', 'keydown'].forEach(event => {
    document.addEventListener(event, enableAudio, { once: true });
  });
});

button.addEventListener('click', () => {
    if (intervalId) return;

    // Use the configurable duration from settings
    totalSeconds = timerDuration * 60;

    // Update button text
    button.textContent = 'Timer Running...';
    button.classList.add('timer-running');

    updateDisplay();

    intervalId = setInterval(() => {
        totalSeconds--;
        updateDisplay();

        if (totalSeconds <= 0) {
            clearInterval(intervalId);
            intervalId = null;

            // Play sound
            ping.currentTime = 0; // Reset to start
            ping.play().catch(err => {
                console.log("Audio play blocked:", err);
                // Visual alert as fallback
                alert("Time's up!");
            });

            // Visual feedback
            timerDisplay.classList.add('timer-complete');
            timerContainer.classList.add('timer-complete');

            // Reset button
            button.textContent = `Start ${timerDuration} Min Timer`;
            button.classList.remove('timer-running');
            button.disabled = false;

            // Reset after 3 seconds
            setTimeout(() => {
                timerDisplay.classList.remove('timer-complete');
                timerContainer.classList.remove('timer-complete');
                totalSeconds = timerDuration * 60;
                updateDisplay();
            }, 3000);
        }
    }, 1000);

    button.disabled = true;
});

function updateDisplay() {
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    timerDisplay.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
}