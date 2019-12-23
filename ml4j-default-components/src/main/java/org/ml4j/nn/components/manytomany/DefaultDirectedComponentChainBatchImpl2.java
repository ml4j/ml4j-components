package org.ml4j.nn.components.manytomany;

import java.util.List;
import java.util.stream.Collectors;

import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;

public class DefaultDirectedComponentChainBatchImpl2 extends DefaultDirectedComponentBatchImpl<DefaultDirectedComponentChain, DefaultDirectedComponentChainActivation> implements
DefaultDirectedComponentChainBatch<DefaultDirectedComponentChain, DefaultDirectedComponentChainActivation>{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public DefaultDirectedComponentChainBatchImpl2(List<DefaultDirectedComponentChain> components) {
		super(components);
	}

	@Override
	public DefaultDirectedComponentChainBatch<DefaultDirectedComponentChain, DefaultDirectedComponentChainActivation> dup() {
		return new DefaultDirectedComponentChainBatchImpl2(this.getComponents().stream().map(c -> c.dup()).collect(Collectors.toList()));
	}

}
