export const formatTimestamp = (seconds: number) => {
  // Ensure the input is a number and round it to remove the decimal part
  seconds = Math.round(Number(seconds));

  // Calculate hours, minutes, and remaining seconds
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainingSeconds = seconds % 60;

  // Format each part to ensure two digits, e.g., '04' instead of '4'
  const formattedMinutes = minutes.toString().padStart(2, "0");
  const formattedSeconds = remainingSeconds.toString().padStart(2, "0");

  // Construct and return the timestamp
  return `${formattedMinutes}:${formattedSeconds}`;
};
