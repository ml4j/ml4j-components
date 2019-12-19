package org.ml4j.nn.components.manytoone;

import java.util.ArrayList;
import java.util.List;

import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.components.manytoone.base.ManyToOneDirectedComponentActivationBase;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DummyManyToOneDirectedComponentActivation extends ManyToOneDirectedComponentActivationBase implements ManyToOneDirectedComponentActivation {

	private static final Logger LOGGER = LoggerFactory.getLogger(DummyManyToOneDirectedComponentActivation.class);
	
	private int size;
	
	public DummyManyToOneDirectedComponentActivation(int size, NeuronsActivation output) {
		super(output);
		this.size = size;
	}
	
	@Override
	public DirectedComponentGradient<List<NeuronsActivation>> backPropagate(
			DirectedComponentGradient<NeuronsActivation> gradient) {
		LOGGER.debug("Mock back propagating single neurons activations into multiple back propagated neurons activations");
		List<NeuronsActivation> activations = new ArrayList<>();
		for (int i = 0; i < size; i++) {
			activations.add(gradient.getOutput());
		}
		return new DirectedComponentGradientImpl<>(activations);
	}

}
