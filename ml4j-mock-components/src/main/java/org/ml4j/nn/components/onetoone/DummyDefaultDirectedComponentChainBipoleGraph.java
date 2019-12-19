package org.ml4j.nn.components.onetoone;

import java.util.Arrays;
import java.util.List;

import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.manytomany.DefaultDirectedComponentChainBatch;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentBipoleGraph;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentBipoleGraphActivation;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.components.onetoone.base.DefaultDirectedComponentChainBipoleGraphBase;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DummyDefaultDirectedComponentChainBipoleGraph extends DefaultDirectedComponentChainBipoleGraphBase implements DefaultDirectedComponentBipoleGraph {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private DefaultDirectedComponentChainBatch<DefaultDirectedComponentChain, DefaultDirectedComponentChainActivation> parallelComponentChainsBatch;

	public DummyDefaultDirectedComponentChainBipoleGraph(
			DefaultDirectedComponentChainBatch<DefaultDirectedComponentChain, DefaultDirectedComponentChainActivation> parallelComponentChainsBatch) {
		super(parallelComponentChainsBatch);
	}

	@Override
	public DefaultDirectedComponentChainBatch<DefaultDirectedComponentChain, DefaultDirectedComponentChainActivation> getEdges() {
		throw new UnsupportedOperationException();
	}
	
	@Override
	public DefaultDirectedComponentBipoleGraphActivation forwardPropagate(NeuronsActivation neuronsActivation,
			DirectedComponentsContext context) {
		return new DummyDefaultDirectedComponentBipoleGraphActivation(neuronsActivation);
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> decompose() {
		return Arrays.asList(this);
	}

	@Override
	public DefaultDirectedComponentBipoleGraph dup() {
		return new DummyDefaultDirectedComponentChainBipoleGraph(parallelComponentChainsBatch.dup());
	}

}
