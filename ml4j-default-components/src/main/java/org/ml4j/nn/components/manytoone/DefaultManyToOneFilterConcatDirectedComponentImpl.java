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
package org.ml4j.nn.components.manytoone;

import java.util.List;
import java.util.Optional;
import java.util.function.Predicate;
import java.util.stream.Collectors;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.ml4j.images.ChannelConcatImages;
import org.ml4j.images.Images;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.manytoone.base.ManyToOneDirectedComponentBase;
import org.ml4j.nn.neurons.ImageNeuronsActivation;
import org.ml4j.nn.neurons.ImageNeuronsActivationImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.ml4j.nn.neurons.format.features.DimensionScope;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.codepoetics.protonpack.StreamUtils;

public class DefaultManyToOneFilterConcatDirectedComponentImpl
		extends ManyToOneDirectedComponentBase<DefaultManyToOneDirectedComponentActivationImpl>
		implements ManyToOneDirectedComponent<DefaultManyToOneDirectedComponentActivationImpl> {

	private Neurons3D outputNeurons;

	public DefaultManyToOneFilterConcatDirectedComponentImpl(Neurons3D outputNeurons) {
		super(PathCombinationStrategy.FILTER_CONCAT);
		this.outputNeurons = outputNeurons;
	}

	private static final Logger LOGGER = LoggerFactory
			.getLogger(DefaultManyToOneFilterConcatDirectedComponentImpl.class);

	/**
	 * s Serialization id.
	 */
	private static final long serialVersionUID = -7049642040068320620L;

	@Override
	public DefaultManyToOneDirectedComponentActivationImpl forwardPropagate(List<NeuronsActivation> neuronActivations,
			DirectedComponentsContext context) {
		
		if (!neuronActivations.stream().map(n -> isSupported(n.getFormat())).allMatch(Predicate.isEqual(true))) {
			throw new IllegalArgumentException("Unsupported NeuronsActivation format");
		}
		
		LOGGER.debug("Combining multiple neurons activations into a single output neurons activation");
		Pair<ImageNeuronsActivation, int[]> b = getCombinedOutput(neuronActivations, context);
		return new ManyToOneFilterConcatDirectedComponentActivation(neuronActivations.size(), b.getRight(),
				b.getLeft());
	}

	private Pair<ImageNeuronsActivation, int[]> getCombinedOutput(List<NeuronsActivation> inputs,
			DirectedComponentsContext context) {

		List<Images> images = inputs.stream()
				.map(i -> i.asImageNeuronsActivation(getInputNeurons3D(i.getNeurons()), DimensionScope.OUTPUT).getImages())
				.collect(Collectors.toList());

		Images result = new ChannelConcatImages(images, outputNeurons.getHeight(), outputNeurons.getWidth(), 0, 0,
				images.get(0).getExamples());

		int[] channelBoundaries = new int[inputs.size()];
		StreamUtils.zipWithIndex(inputs.stream().map(i -> getInputNeurons3D(i.getNeurons()).getDepth())).forEach(e -> {
			if (e.getIndex() == 0) {
				channelBoundaries[(int) e.getIndex()] = e.getValue();
			} else {
				channelBoundaries[(int) e.getIndex()] = channelBoundaries[(int) e.getIndex() - 1] + e.getValue();
			}
		});

		LOGGER.debug("End Combining input for many to one junction:" + result.getChannels());

		return new ImmutablePair<>(new ImageNeuronsActivationImpl(
				new Neurons3D(outputNeurons.getWidth(), outputNeurons.getHeight(), result.getChannels(), false), result,
				inputs.get(0).asImageNeuronsActivation(getInputNeurons3D(inputs.get(0).getNeurons()), DimensionScope.OUTPUT).getFormat(), false), channelBoundaries);
	}

	private Neurons3D getInputNeurons3D(Neurons inputNeurons) {
		int inputNeuronsFeatureCount = inputNeurons.getNeuronCountExcludingBias();
		int calculatedInputNeuronsDepth = inputNeuronsFeatureCount
				/ (outputNeurons.getWidth() * outputNeurons.getHeight());
		if (inputNeuronsFeatureCount != calculatedInputNeuronsDepth * outputNeurons.getWidth()
				* outputNeurons.getHeight()) {
			throw new IllegalArgumentException(
					"One of the inputs to many to one component cannot be converted to Neurons3D of the correct dimensions");
		}
		return new Neurons3D(outputNeurons.getWidth(), outputNeurons.getHeight(), calculatedInputNeuronsDepth,
				inputNeurons.hasBiasUnit());
	}

	@Override
	public ManyToOneDirectedComponent<DefaultManyToOneDirectedComponentActivationImpl> dup() {
		return new DefaultManyToOneFilterConcatDirectedComponentImpl(outputNeurons);
	}

	@Override
	public Optional<NeuronsActivationFormat<?>> optimisedFor() {
		return Optional.empty();
	}

	@Override
	public boolean isSupported(NeuronsActivationFormat<?> format) {
		return NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET.equals(format.getFeatureOrientation());
	}
}
