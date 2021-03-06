package org.ml4j.nn.axons;

import java.util.function.Supplier;

import org.ml4j.nn.neurons.NeuronsActivation;

public class NoOpAxonsActivation implements AxonsActivation {

	private Axons<?, ?, ?> axons;
	private Supplier<NeuronsActivation> input;
	private NeuronsActivation output;

	public NoOpAxonsActivation(Axons<?, ?, ?> axons, Supplier<NeuronsActivation> input, NeuronsActivation output) {
		this.axons = axons;
		this.input = input;
		this.output = output;
	}

	@Override
	public Axons<?, ?, ?> getAxons() {
		return axons;
	}

	@Override
	public AxonsDropoutMask getDropoutMask() {
		// No dropout mask by default
		return null;
	}

	@Override
	public NeuronsActivation getPostDropoutOutput() {
		return output;
	}

	@Override
	public Supplier<NeuronsActivation> getPostDropoutInput() {
		return input;
	}

	@Override
	public void setDropoutMask(AxonsDropoutMask axonsDropoutMask) {
		// no-op
	}
}
