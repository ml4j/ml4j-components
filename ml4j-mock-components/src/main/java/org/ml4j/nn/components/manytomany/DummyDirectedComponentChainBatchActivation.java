package org.ml4j.nn.components.manytomany;

import java.util.List;

import org.ml4j.nn.components.DirectedComponentBatchActivation;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.manytomany.base.DirectedComponentChainBatchActivationBase;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DummyDirectedComponentChainBatchActivation extends DirectedComponentChainBatchActivationBase implements DirectedComponentBatchActivation<NeuronsActivation, DefaultDirectedComponentChainActivation>{

	
	public DummyDirectedComponentChainBatchActivation(List<DefaultDirectedComponentChainActivation> activations) {
		super(activations);
	}

	@Override
	public DirectedComponentGradient<List<NeuronsActivation>> backPropagate(
			DirectedComponentGradient<List<NeuronsActivation>> gradient) {
		return gradient;
	}
}
