function loadImage() {
    setButtonDisabled(true);
    const sliderValues = getSliderValues();
    const apiPostOptions = {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(sliderValues)
    };
    fetch("/api", apiPostOptions)
        .then(response => response.blob())
        .then(blob => {
            console.log(blob);
            const blobUrl = URL.createObjectURL(blob);
            console.log(blobUrl);
            document.getElementById("demo-image").src = blobUrl;
            setButtonDisabled(false);
        });
}

function setButtonDisabled(disabledProp) {
    const DISABLED_STYLING = ' opacity-50 cursor-not-allowed';
    const button = document.getElementById('generate-button');
    button.disabled = disabledProp;
    button.className = disabledProp ? button.className.concat(DISABLED_STYLING) : button.className.replace(DISABLED_STYLING, ''); 
}

function getSliderValues() {
    const userInputs = document.getElementsByTagName('input');
    return Array.from(userInputs).map(input => input.value);
}

function handleSliderChange() {
    const sliderValues = getSliderValues();
    const valueLabels = document.getElementsByClassName('slider-value');
    Array.from(valueLabels).forEach((valueLabel, index) => valueLabel.innerHTML = sliderValues[index]);
}