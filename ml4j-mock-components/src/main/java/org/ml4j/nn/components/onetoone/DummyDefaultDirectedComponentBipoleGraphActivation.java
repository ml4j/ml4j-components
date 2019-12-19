package org.ml4j.nn.components.onetoone;

import java.util.Arrays;
import java.util.List;

import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentBipoleGraphActivation;
import org.ml4j.nn.components.onetoone.base.DefaultDirectedComponentBipoleGraphActivationBase;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DummyDefaultDirectedComponentBipoleGraphActivation extends DefaultDirectedComponentBipoleGraphActivationBase
		implements DefaultDirectedComponentBipoleGraphActivation {
	
	public DummyDefaultDirectedComponentBipoleGraphActivation(NeuronsActivation output) {
		super(output);
	}
	
	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(
			DirectedComponentGradient<NeuronsActivation> gradient) {
		return gradient;
	}

	@Override
	public List<DefaultChainableDirectedComponentActivation> decompose() {
		return Arrays.asList(this);
	}

}
