package org.ml4j.nn.axons;

import org.ml4j.nn.axons.base.AxonsBase;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DummyAxons<L extends Neurons, R extends Neurons> extends AxonsBase<L, R,  DummyAxons<L, R>> implements Axons<L, R, DummyAxons<L, R>> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	public DummyAxons(L leftNeurons, R rightNeurons) {
		super(leftNeurons, rightNeurons);
	}

	@Override
	public DummyAxons<L, R> dup() {
		return new DummyAxons<>(leftNeurons, rightNeurons);
	}

	@Override
	public boolean isTrainable(AxonsContext axonsContext) {
		return true;
	}

	@Override
	public AxonsActivation pushLeftToRight(NeuronsActivation arg0, AxonsActivation arg1, AxonsContext arg2) {
		throw new UnsupportedOperationException();
	}

	@Override
	public AxonsActivation pushRightToLeft(NeuronsActivation arg0, AxonsActivation arg1, AxonsContext arg2) {
		throw new UnsupportedOperationException();
	}

}
