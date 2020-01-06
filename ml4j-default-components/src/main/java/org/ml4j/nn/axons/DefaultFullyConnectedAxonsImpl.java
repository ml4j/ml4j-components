package org.ml4j.nn.axons;

import org.ml4j.nn.neurons.Neurons;

public class DefaultFullyConnectedAxonsImpl extends WeightedAxonsBase<Neurons, Neurons, FullyConnectedAxons> implements FullyConnectedAxons {
	
	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	public DefaultFullyConnectedAxonsImpl(Neurons leftNeurons, Neurons rightNeurons, AxonWeights axonWeights) {
		super(leftNeurons, rightNeurons, axonWeights);
	}

	@Override
	public FullyConnectedAxons dup() {
		return new DefaultFullyConnectedAxonsImpl(leftNeurons, rightNeurons, axonWeights.dup());
	}

	@Override
	protected boolean isLeftInputDropoutSupported() {
		return true;
	}
}
