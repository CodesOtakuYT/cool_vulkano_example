use bytemuck::{Pod, Zeroable};
use core::cmp::{max, min};
use std::sync::Arc;
use std::time::Instant;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassContents,
    },
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
    },
    image::{view::ImageView, ImageAccess, ImageUsage, SwapchainImage},
    impl_vertex,
    instance::{Instance, InstanceCreateInfo},
    pipeline::{
        graphics::{
            input_assembly::InputAssemblyState,
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    swapchain::{
        acquire_next_image, AcquireError, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
    },
    sync::{self, FlushError, GpuFuture},
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

fn main() {
    let app_start = Instant::now();

    let background_color = [0.1, 0.1, 0.1, 1.0];
    let swapchain_buffers_count = 3; // triple buffering
    let instance_count = 1000;

    let required_extensions = vulkano_win::required_extensions();

    let instance = Instance::new(InstanceCreateInfo {
        enabled_extensions: required_extensions,
        enumerate_portability: true,
        ..Default::default()
    })
    .unwrap();

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };

    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&physical_device| {
            physical_device
                .supported_extensions()
                .is_superset_of(&device_extensions)
        })
        .filter_map(|physical_device| {
            physical_device
                .queue_families()
                .find(|&queue_family| {
                    queue_family.supports_graphics()
                        && queue_family.supports_surface(&surface).unwrap_or(false)
                })
                .map(|queue_family| (physical_device, queue_family))
        })
        .min_by_key(
            |(physical_device, _)| match physical_device.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
            },
        )
        .unwrap();

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    let (logical_device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let surface_capabilities = physical_device
            .surface_capabilities(&surface, Default::default())
            .unwrap();

        println!(
            "Swapchain buffers count: {}/{:?}",
            swapchain_buffers_count,
            surface_capabilities.max_image_count.unwrap_or(0)
        );

        let image_format = Some(
            physical_device
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0,
        );

        let min_image_count = match surface_capabilities.max_image_count {
            None => max(
                swapchain_buffers_count,
                surface_capabilities.min_image_count,
            ),
            Some(limit) => min(
                max(
                    swapchain_buffers_count,
                    surface_capabilities.min_image_count,
                ),
                limit,
            ),
        };

        Swapchain::new(
            logical_device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count,
                image_format,
                image_extent: surface.window().inner_size().into(),
                image_usage: ImageUsage::color_attachment(),
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .iter()
                    .next()
                    .unwrap(),
                ..Default::default()
            },
        )
        .unwrap()
    };

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
    struct Vertex {
        position: [f32; 2],
        color: [f32; 4],
    }
    impl_vertex!(Vertex, position, color);

    let vertices = [
        Vertex {
            position: [-0.5, -0.25],
            color: [1.0, 0.0, 0.0, 1.0],
        },
        Vertex {
            position: [0.0, 0.5],
            color: [0.0, 1.0, 0.0, 1.0],
        },
        Vertex {
            position: [0.25, -0.1],
            color: [0.0, 0.0, 1.0, 1.0],
        },
    ];

    let vertex_buffer =
        CpuAccessibleBuffer::from_iter(logical_device.clone(), BufferUsage::all(), false, vertices)
            .unwrap();

    mod vertex_shader {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: "
            #version 460

            layout(location = 0) in vec2 position;
            layout(location = 1) in vec4 color;
            
            layout(location = 0) out vec4 out_color;

            layout(push_constant) uniform PushConstantData {
                float time;
                float x;
                float y;
            } pc;

            void main() {
                out_color = color;
                float time = pc.time;
                float mouse_x = pc.x;
                float mouse_y = pc.y;
                int x = gl_InstanceIndex;
                vec2 pos = position*vec2(mouse_x, mouse_y);
                gl_Position = vec4(pos+vec2(sin(time+position.x+position.y+x)*0.5, sin(time+position.x+position.y+x*2)*0.5), 0.0, 1.0);
            }
            "
        }
    }

    mod fragment_shader {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: "
            #version 460

            layout(location = 0) out vec4 f_color;
            layout(location = 0) in vec4 in_color;

            void main() {
                f_color = in_color;
            }
            "
        }
    }

    let loaded_vertex_shader = vertex_shader::load(logical_device.clone()).unwrap();
    let loaded_fragment_shader = fragment_shader::load(logical_device.clone()).unwrap();

    let render_pass = vulkano::single_pass_renderpass!(
        logical_device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.image_format(),
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    )
    .unwrap();

    let graphics_pipeline = GraphicsPipeline::start()
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .input_assembly_state(InputAssemblyState::new())
        .vertex_shader(loaded_vertex_shader.entry_point("main").unwrap(), ())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(loaded_fragment_shader.entry_point("main").unwrap(), ())
        .build(logical_device.clone())
        .unwrap();

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };

    let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut viewport);

    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(sync::now(logical_device.clone()).boxed());

    use winit::dpi::PhysicalPosition;
    let mut mouse_pos = [0.0, 0.0];

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CursorMoved { position, .. },
            ..
        } => {
            let dimensions = surface.window().inner_size();
            mouse_pos = [position.x/(dimensions.width as f64), position.y/(dimensions.height as f64)];
        }
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => {
            recreate_swapchain = true;
        }
        Event::RedrawEventsCleared => {
            let dimensions = surface.window().inner_size();
            if dimensions.width == 0 || dimensions.height == 0 {
                return;
            }
            previous_frame_end.as_mut().unwrap().cleanup_finished();
            if recreate_swapchain {
                let (new_swapchain, new_images) = match swapchain.recreate(SwapchainCreateInfo {
                    image_extent: dimensions.into(),
                    ..swapchain.create_info()
                }) {
                    Ok(r) => r,
                    Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                    Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                };
                swapchain = new_swapchain;
                framebuffers =
                    window_size_dependent_setup(&new_images, render_pass.clone(), &mut viewport);
                recreate_swapchain = false;
            }

            let (image_num, suboptimal, acquire_future) =
                match acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("Failed to acquire next image: {:?}", e),
                };

            if suboptimal {
                recreate_swapchain = true;
            }

            let mut builder = AutoCommandBufferBuilder::primary(
                logical_device.clone(),
                queue.family(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            let push_constants = vertex_shader::ty::PushConstantData {
                time: (Instant::now() - app_start).as_secs_f32(),
                x: mouse_pos[0] as f32,
                y: mouse_pos[1] as f32
            };

            use vulkano::pipeline::Pipeline;
            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![Some(background_color.into())],
                        ..RenderPassBeginInfo::framebuffer(framebuffers[image_num].clone())
                    },
                    SubpassContents::Inline,
                )
                .unwrap()
                .set_viewport(0, [viewport.clone()])
                .bind_pipeline_graphics(graphics_pipeline.clone())
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .push_constants(graphics_pipeline.layout().clone(), 0, push_constants)
                .draw(vertex_buffer.len() as u32, instance_count, 0, 0)
                .unwrap()
                .end_render_pass()
                .unwrap();
            let command_buffer = builder.build().unwrap();

            let future = previous_frame_end
                .take()
                .unwrap()
                .join(acquire_future)
                .then_execute(queue.clone(), command_buffer)
                .unwrap()
                .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                .then_signal_fence_and_flush();

            match future {
                Ok(future) => {
                    previous_frame_end = Some(future.boxed());
                }
                Err(FlushError::OutOfDate) => {
                    recreate_swapchain = true;
                    previous_frame_end = Some(sync::now(logical_device.clone()).boxed());
                }
                Err(e) => {
                    println!("Failed to flush future: {:?}", e);
                    previous_frame_end = Some(sync::now(logical_device.clone()).boxed());
                }
            }
        }
        _ => (),
    });
}

fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> Vec<Arc<Framebuffer>> {
    let dimensions = images[0].dimensions().width_height();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}
