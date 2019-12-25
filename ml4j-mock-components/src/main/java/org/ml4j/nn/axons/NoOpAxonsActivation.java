package org.ml4j.nn.axons;

import org.ml4j.nn.neurons.NeuronsActivation;

public class NoOpAxonsActivation implements AxonsActivation {
	
	private Axons<?, ?, ?> axons;
	private NeuronsActivation input;
	private NeuronsActivation output;
	
	public NoOpAxonsActivation(Axons<?, ?, ?> axons, NeuronsActivation input, NeuronsActivation output) {
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
	public NeuronsActivation getPostDropoutInput() {
		return input;
	}
}
