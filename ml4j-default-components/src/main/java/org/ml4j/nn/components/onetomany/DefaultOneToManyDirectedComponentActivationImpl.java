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
package org.ml4j.nn.components.onetomany;

import java.util.List;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.components.onetomany.base.OneToManyDirectedComponentActivationBase;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default implementation of OneToManyDirectedComponentActivation -
 * encapsulating the activation from a DefaultOneToManyDirectedComponent.
 * 
 * @author Michael Lavelle
 */
public class DefaultOneToManyDirectedComponentActivationImpl extends OneToManyDirectedComponentActivationBase
		implements OneToManyDirectedComponentActivation {

	private static final Logger LOGGER = LoggerFactory.getLogger(DefaultOneToManyDirectedComponentActivationImpl.class);

	private MatrixFactory matrixFactory;
	private Neurons inputNeurons;

	/**
	 * DefaultOneToManyDirectedComponentActivationImpl constructor
	 * 
	 * @param input                        The neurons activation input to the one
	 *                                     to many component.
	 * @param outputNeuronsActivationCount The desired number of instances of output
	 *                                     neuron activations, one for each of the
	 *                                     components on the RHS of the
	 *                                     OneToManyDirectedComponentActivation.
	 */
	public DefaultOneToManyDirectedComponentActivationImpl(MatrixFactory matrixFactory, NeuronsActivation input,
			int outputNeuronsActivationCount) {
		super(input, outputNeuronsActivationCount);
		this.matrixFactory = matrixFactory;
		this.inputNeurons = input.getNeurons();
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(
			DirectedComponentGradient<List<NeuronsActivation>> gradient) {
		LOGGER.debug(
				"Back propagating multiple gradient neurons activations into a single combined neurons activation");
		List<NeuronsActivation> gradients = gradient.getOutput();

		NeuronsActivation totalActivation = new NeuronsActivationImpl(inputNeurons,
				matrixFactory.createMatrix(gradients.get(0).getRows(), gradients.get(0).getColumns()),
				gradients.get(0).getFeatureOrientation(), false);

		for (NeuronsActivation activation : gradients) {
			totalActivation.addInline(matrixFactory, activation);
			activation.close();
		}
		return new DirectedComponentGradientImpl<>(gradient.getTotalTrainableAxonsGradients(), totalActivation);
	}
}
