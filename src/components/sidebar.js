import headshot from "../content/prof.jpg";
import React, { Component } from "react";

class Sidebar extends Component {

	render() {
		return (
			<div id="sidebar">
				<div id="profile">
					<h2>Akshat Pandey</h2>
					<img src={headshot} alt="" />
					<div id="media">
						<ul class="nobullet">
							<li><a href="https://linkedin.com/in/akshatpandey" target="_blank" rel="noopener noreferrer">Linkedin</a></li>
							<li><a href="https://github.com/akshat0123" target="_blank" rel="noopener noreferrer">Github</a></li>
						</ul>
					</div>
				</div>
			</div>
		);
	}

}

export default Sidebar
