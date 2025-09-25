import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

import viser
import viser.transforms as tf

from src.flame import FLAME


@dataclass
class GuiElements:
    """Structure containing handles for reading from GUI elements."""
    mesh_color: viser.GuiInputHandle[Tuple[int, int, int]]
    edges_color: viser.GuiInputHandle[Tuple[int, int, int]]
    points_color: viser.GuiInputHandle[Tuple[int, int, int]]
    mesh_visibility: viser.GuiInputHandle[bool]
    edges_visibility: viser.GuiInputHandle[bool]
    points_visibility: viser.GuiInputHandle[bool]
    points_radius: viser.GuiInputHandle[float]
    gui_shapes: List[viser.GuiInputHandle[float]]
    gui_expressions: List[viser.GuiInputHandle[float]]
    gui_joints: List[viser.GuiInputHandle[Tuple[float, float, float]]]
    joints_controls: List[viser.TransformControlsHandle]
    is_changed: bool


def create_gui_elements(
    server: viser.ViserServer,
    num_shapes: int,
    num_expressions: int,
    num_joints: int,
    joints_parents: np.ndarray,
    joints_desc: dict
) -> GuiElements:

    tab_group = server.gui.add_tab_group()

    def set_changed(_) -> None:
        out.is_changed = True

    # GUI elements: mesh settings + visibility.
    with tab_group.add_tab("View", viser.Icon.VIEWFINDER):
        mesh_color = server.gui.add_rgb("Color", initial_value=(90, 200, 255))
        edges_color = server.gui.add_rgb("Edges Color", initial_value=(0, 0, 0))
        points_color = server.gui.add_rgb("Points Color", initial_value=(255, 0, 0))
        mesh_visibility = server.gui.add_checkbox("Mesh Visibility", initial_value=True)
        edges_visibility = server.gui.add_checkbox("Edges Visibility", initial_value=True)
        points_visibility = server.gui.add_checkbox("Points Visibility", initial_value=True)
        points_radius = server.gui.add_slider("Points Radius", min=0.0001, max=0.001, step=0.00001, initial_value=0.0005)
        gui_show_controls = server.gui.add_checkbox("Handles", initial_value=True)
        gui_control_size = server.gui.add_slider("Handle size", min=0.0, max=1.0, step=0.01, initial_value=0.5)

        mesh_color.on_update(set_changed)
        edges_color.on_update(set_changed)
        points_color.on_update(set_changed)
        mesh_visibility.on_update(set_changed)
        edges_visibility.on_update(set_changed)
        points_visibility.on_update(set_changed)
        points_radius.on_update(set_changed)

        @gui_show_controls.on_update
        def _(_):
            for control in joints_controls:
                control.visible = gui_show_controls.value

        @gui_control_size.on_update
        def _(_):
            for control in joints_controls:
                control.scale = gui_control_size.value / 20.0

    # GUI elements: shape parameters.
    with tab_group.add_tab("Shape", viser.Icon.BOX):
        gui_reset_shape = server.gui.add_button("Reset Shape")
        gui_random_shape = server.gui.add_button("Random Shape")

        @gui_reset_shape.on_click
        def _(_):
            for coeff in gui_shapes:
                coeff.value = 0.0

        @gui_random_shape.on_click
        def _(_):
            for coeff in gui_shapes:
                coeff.value = np.random.normal(loc=0.0, scale=1.0)

        gui_shapes = []
        for i in range(num_shapes):
            coeff = server.gui.add_slider(
                f"Shape #{i + 1}", min=-3.0, max=3.0, step=0.01, initial_value=0.0
            )
            gui_shapes.append(coeff)
            coeff.on_update(set_changed)

    # GUI elements: expression parameters.
    with tab_group.add_tab("Expr", viser.Icon.ARROW_LEFT_RIGHT):
        gui_reset_expression = server.gui.add_button("Reset Expression")
        gui_random_expression = server.gui.add_button("Random Expression")

        @gui_reset_expression.on_click
        def _(_):
            for coeff in gui_expressions:
                coeff.value = 0.0

        @gui_random_expression.on_click
        def _(_):
            for coeff in gui_expressions:
                coeff.value = np.random.normal(loc=0.0, scale=1.0)

        gui_expressions = []
        for i in range(num_expressions):
            coeff = server.gui.add_slider(
                f"Expr #{i + 1}", min=-3.0, max=3.0, step=0.01, initial_value=0.0
            )
            gui_expressions.append(coeff)
            coeff.on_update(set_changed)

    # GUI elements: joint angles.
    with tab_group.add_tab("Joints", viser.Icon.ANGLE):
        gui_reset_joints = server.gui.add_button("Reset Joints")
        gui_random_joints = server.gui.add_button("Random Joints")

        @gui_reset_joints.on_click
        def _(_):
            for joint in gui_joints:
                joint.value = (0.0, 0.0, 0.0)

        @gui_random_joints.on_click
        def _(_):
            rng = np.random.default_rng()
            for joint in gui_joints:
                joint.value = tf.SO3.sample_uniform(rng).log()

        gui_joints: List[viser.GuiInputHandle[Tuple[float, float, float]]] = []
        for i in range(num_joints):
            gui_joint = server.gui.add_vector3(
                label=joints_desc[i],
                initial_value=(0.0, 0.0, 0.0),
                step=0.05
            )
            gui_joints.append(gui_joint)

            def set_callback_in_closure(i: int) -> None:
                @gui_joint.on_update
                def _(_):
                    joints_controls[i].wxyz = tf.SO3.exp(np.array(gui_joints[i].value)).wxyz
                    out.is_changed = True

            set_callback_in_closure(i)

    # Transform control gizmos on joints.
    joints_controls: List[viser.TransformControlsHandle] = []
    prefixed_joint_names: List[str] = []  # Joint names, but prefixed with parents.
    for i in range(num_joints):
        prefixed_joint_name = joints_desc[i]
        if i > 0:
            prefixed_joint_name = prefixed_joint_names[joints_parents[i]] + "/" + prefixed_joint_name
        prefixed_joint_names.append(prefixed_joint_name)
        controls = server.scene.add_transform_controls(
            f"{prefixed_joint_name}",
            depth_test=False,
            scale=0.025,
            disable_axes=True,
            disable_sliders=True,
            visible=gui_show_controls.value
        )
        joints_controls.append(controls)

        def set_callback_in_closure(i: int) -> None:
            @controls.on_update
            def _(_) -> None:
                axisangle = tf.SO3(joints_controls[i].wxyz).log()
                gui_joints[i].value = (axisangle[0], axisangle[1], axisangle[2])

        set_callback_in_closure(i)

    out = GuiElements(
        mesh_color=mesh_color,
        edges_color=edges_color,
        points_color=points_color,
        mesh_visibility=mesh_visibility,
        edges_visibility=edges_visibility,
        points_visibility=points_visibility,
        points_radius=points_radius,
        gui_shapes=gui_shapes,
        gui_expressions=gui_expressions,
        gui_joints=gui_joints,
        joints_controls=joints_controls,
        is_changed=True
    )
    return out


def main(control_eyelids: bool = False) -> None:
    server = viser.ViserServer()
    server.scene.set_up_direction("+y")
    server.gui.configure_theme(control_layout="collapsible")

    # Main loop
    model = FLAME(control_eyelids=control_eyelids)
    gui_elements = create_gui_elements(
        server,
        num_shapes=model.num_shapes,
        num_expressions=model.num_expressions,
        num_joints=model.num_joints,
        joints_parents=model.parents,
        joints_desc=model.JOINT_DESC
    )
    vertices, _ = model.run()
    mesh_handle = server.scene.add_mesh_simple(
        "/head",
        vertices=vertices,
        faces=model.faces_numpy,
        wireframe=False,
        color=gui_elements.mesh_color.value
    )
    edges_handle = server.scene.add_mesh_simple(
        "/edges",
        vertices=vertices,
        faces=model.faces_numpy,
        wireframe=True,
        color=gui_elements.edges_color.value
    )
    points_handle = server.scene.add_point_cloud(
        "/points",
        points=vertices,
        colors=np.tile(np.array(gui_elements.points_color.value) / 255.0, (len(vertices), 1)),
        point_size=gui_elements.points_radius.value,
        point_shape="circle"
    )

    while True:
        time.sleep(0.005)
        if not gui_elements.is_changed:
            continue

        if gui_elements.is_changed:
            vertices, joints = model.run(
                shape=np.array([x.value for x in gui_elements.gui_shapes]),
                expression=np.array([x.value for x in gui_elements.gui_expressions]),
                pose=np.array(gui_elements.gui_joints[0].value),
                neck=np.array(gui_elements.gui_joints[1].value),
                jaw=np.array(gui_elements.gui_joints[2].value),
                left_eye=np.array(gui_elements.gui_joints[3].value),
                right_eye=np.array(gui_elements.gui_joints[4].value)
            )
            mesh_handle.vertices = vertices
            edges_handle.vertices = vertices
            points_handle.points = vertices

        gui_elements.is_changed = False
        mesh_handle.visible = gui_elements.mesh_visibility.value
        mesh_handle.color = gui_elements.mesh_color.value
        edges_handle.color = gui_elements.edges_color.value
        points_handle.colors = np.tile(np.array(gui_elements.points_color.value) / 255.0, (len(vertices), 1))
        edges_handle.visible = gui_elements.edges_visibility.value
        points_handle.visible = gui_elements.points_visibility.value
        points_handle.point_size = gui_elements.points_radius.value

        for i, control in enumerate(gui_elements.joints_controls):
            control.position = joints[i]
