package org.ml4j.nn.activationfunctions.mocks;

import java.util.Optional;

import org.ml4j.nn.activationfunctions.ActivationFunctionProperties;
import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContext;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;

public class DummyDifferentiableActivationFunctionImpl implements DifferentiableActivationFunction {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private ActivationFunctionType activationFunctionType;

	private boolean setOutputToZeros;

	public DummyDifferentiableActivationFunctionImpl(ActivationFunctionType activationFunctionType,
			boolean setOutputToZeros) {
		this.activationFunctionType = activationFunctionType;
		this.setOutputToZeros = setOutputToZeros;
	}

	@Override
	public DifferentiableActivationFunctionActivation activate(NeuronsActivation activation,
			NeuronsActivationContext context) {
		NeuronsActivation output = activation;
		float[] values = output.getActivations(context.getMatrixFactory()).getRowByRowArray();
		if (setOutputToZeros)
			output.applyValueModifier(v -> v == values[0] ? -1 : 0);
		return new DummyDifferentiableActivationFunctionActivationImpl(this, activation, output);
	}

	@Override
	public ActivationFunctionType getActivationFunctionType() {
		return activationFunctionType;
	}

	@Override
	public NeuronsActivation activationGradient(DifferentiableActivationFunctionActivation activation,
			NeuronsActivationContext context) {
		return activation.getInput().dup();
	}

	@Override
	public Optional<NeuronsActivationFormat<?>> optimisedFor() {
		return Optional.empty();
	}

	@Override
	public ActivationFunctionProperties getActivationFunctionProperties() {
		return new ActivationFunctionProperties();
	}
	
	@Override
	public boolean isSupported(NeuronsActivationFormat<?> format) {
		return NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET.equals(format.getFeatureOrientation());
	}
}
