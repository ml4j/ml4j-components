package org.ml4j.nn.axons.mocks;

import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.AxonsDropoutMask;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DummyAxonsActivation implements AxonsActivation {

	private Axons<?, ?, ?> axons;
	private NeuronsActivation inputActivation;
	private NeuronsActivation outputActivation;
	
	public DummyAxonsActivation(Axons<?, ?, ?> axons, NeuronsActivation inputActivation,
			NeuronsActivation outputActivation) {
		this.axons = axons;
		this.inputActivation = inputActivation;
		this.outputActivation = outputActivation;
	}

	@Override
	public Axons<?, ?, ?> getAxons() {
		return axons;
	}

	@Override
	public AxonsDropoutMask getDropoutMask() {
		return null;
	}

	@Override
	public NeuronsActivation getPostDropoutOutput() {
		return outputActivation;
	}

	@Override
	public NeuronsActivation getPostDropoutInput() {
		return inputActivation;
	}
}
