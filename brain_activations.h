static float relu_activation(float value)
{
	T("in=%04.02f\n", value);

	float out = fmax(0.0, value);

	T("out=%04.02f\n", out);

	return(out);
}

static float relu_derivative(float value)
{
	T("in=%04.02f\n", value);

	float out = ((value > 0.0) ? 1.0 : 0.0);

	T("out=%04.02f\n", out);

	return(out);
}

static float sigmoid_activation(float value)
{
	T("in=%04.02f\n", value);
	
	float out = 1.0 / (1.0 + exp(-value));

	T("out=%04.02f\n", out);

	return(out);
}

static float sigmoid_derivative(float value)
{
	T("in=%04.02f\n", value);
	
	float out = value * (1.0 - value);
	
	T("out=%04.02f\n", out);

	return(out);
}
