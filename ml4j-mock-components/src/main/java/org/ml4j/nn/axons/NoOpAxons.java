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
package org.ml4j.nn.axons;

import java.util.Optional;

import org.ml4j.nn.axons.base.AxonsBase;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;

/**
 * Mock implementation of Axons provided for component construction purposes but
 * not intended to be interacted with.
 * 
 * @author Michael Lavelle
 *
 * @param <L> The type of Neurons on the LHS of these Axons
 * @param <R> The type of Neurons on the RHS of these Axons
 * @param <A> The type of Axons.
 */
public class NoOpAxons<L extends Neurons, R extends Neurons> extends AxonsBase<L, R, NoOpAxons<L, R>>
		implements Axons<L, R, NoOpAxons<L, R>> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	public NoOpAxons(AxonsType axonsType, L leftNeurons, R rightNeurons) {
		super(axonsType, leftNeurons, rightNeurons);
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

	@Override
	public Optional<NeuronsActivationFormat<?>> optimisedFor() {
		return Optional.empty();
	}

	@Override
	public boolean isSupported(NeuronsActivationFormat<?> format) {
		return true;
	}


	@Override
	public NoOpAxons<L, R> dup() {
		return new NoOpAxons<>(axonsType, leftNeurons, rightNeurons);
	}
}
