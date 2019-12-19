package org.ml4j.nn.components.onetoone;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.components.onetoone.base.DefaultDirectedComponentChainBase;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DummyDefaultDirectedComponentChain extends DefaultDirectedComponentChainBase implements DefaultDirectedComponentChain {

	/**
	 * Default serialization id
	 */
	private static final long serialVersionUID = 1L;
	
	private static final Logger LOGGER = LoggerFactory.getLogger(DummyDefaultDirectedComponentChain.class);

	public DummyDefaultDirectedComponentChain(List<DefaultChainableDirectedComponent<?, ?>> sequentialComponents) {
		super(sequentialComponents);
	}

	@Override
	public DefaultDirectedComponentChainActivation forwardPropagate(NeuronsActivation neuronsActivation,
			DirectedComponentsContext context) {
		LOGGER.debug("Forward propagating through DummyDirectedComponentChain");
		NeuronsActivation inFlightActivation = neuronsActivation;
		List<DefaultChainableDirectedComponentActivation> activations = new ArrayList<>();
		int index = 0;
		for (DefaultChainableDirectedComponent<?, ?> component : sequentialComponents) {
			DefaultChainableDirectedComponentActivation activation = forwardPropagate(inFlightActivation, component, index, context);
			activations.add(activation);
			inFlightActivation = activation.getOutput();
			index++;
		}
		
		
		return new DummyDefaultDirectedComponentChainActivation(inFlightActivation);
	}

	@Override
	public DefaultDirectedComponentChain dup() {
		return new DummyDefaultDirectedComponentChain(sequentialComponents.stream().map(c -> c.dup()).collect(Collectors.toList()));
	}
}
