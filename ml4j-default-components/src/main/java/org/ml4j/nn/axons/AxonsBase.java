package org.ml4j.nn.axons;

import org.ml4j.nn.neurons.Neurons;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class AxonsBase<L extends Neurons, R extends Neurons, A extends Axons<L, R, A>, C extends AxonsConfig<L, R>>
		implements Axons<L, R, A> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	@SuppressWarnings("unused")
	private static final Logger LOGGER = LoggerFactory.getLogger(AxonsBase.class);

	protected C axonsConfig;

	public AxonsBase(C axonsConfig) {
		super();
		this.axonsConfig = axonsConfig;
	}

	@Override
	public L getLeftNeurons() {
		return axonsConfig.getLeftNeurons();
	}

	@Override
	public R getRightNeurons() {
		return axonsConfig.getRightNeurons();
	}
}
