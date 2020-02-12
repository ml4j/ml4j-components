/*
 * Copyright 2019 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */
package org.ml4j.nn.components.axons;

import java.util.Optional;

import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.NoOpAxonsActivation;
import org.ml4j.nn.components.axons.base.DirectedAxonsComponentBase;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.neurons.DummyNeuronsActivation;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DummyDirectedAxonsComponent<L extends Neurons, R extends Neurons>
		extends DirectedAxonsComponentBase<L, R, Axons<? extends L, ? extends R, ?>>
		implements DirectedAxonsComponent<L, R, Axons<? extends L, ? extends R, ?>> {

	private static final Logger LOGGER = LoggerFactory.getLogger(DummyDirectedAxonsComponent.class);

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	public DummyDirectedAxonsComponent(String name, Axons<? extends L, ? extends R, ?> axons) {
		super(name, axons);
	}

	@Override
	public DirectedAxonsComponentActivation forwardPropagate(NeuronsActivation neuronsActivation,
			AxonsContext context) {
		LOGGER.debug("Forward propagating through DummyDirectedAxonsComponent");

		if (neuronsActivation.getFeatureCount() != this.getInputNeurons().getNeuronCountExcludingBias()) {
			throw new IllegalArgumentException();
		}

		NeuronsActivation dummyOutput = new DummyNeuronsActivation(axons.getRightNeurons(),
				neuronsActivation.getFeatureOrientation(), neuronsActivation.getExampleCount());
		if (dummyOutput.getFeatureCount() != getOutputNeurons().getNeuronCountExcludingBias()) {
			throw new IllegalArgumentException();
		}
		return new DummyDirectedAxonsComponentActivation<>(this,
				new NoOpAxonsActivation(axons, () -> neuronsActivation, dummyOutput), context);
	}

	@Override
	public DirectedAxonsComponent<L, R, Axons<? extends L, ? extends R, ?>> dup(DirectedComponentFactory directedComponentFactory) {
		return new DummyDirectedAxonsComponent<>(name, axons.dup());
	}

	@Override
	public Optional<NeuronsActivationFormat<?>> optimisedFor() {
		return axons.optimisedFor();
	}
	
	@Override
	public boolean isSupported(NeuronsActivationFormat<?> format) {
		return axons.isSupported(format);
	}
}
