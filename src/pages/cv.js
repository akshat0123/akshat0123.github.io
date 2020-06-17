import Sidebar from "../components/sidebar";
import Header from "../components/header";
import Layout from "../components/layout";
import CVData from "../content/cv.json";
import Typist from "react-typist";
import React from "react";

export default () => (
	<div>
		<Header/>
		<Layout>
			<div>
				// <h1><Typist cursor={{hideWhenDone:true, hideWhenDoneDelay:250}}>CV</Typist></h1>
				<h1>CV</h1>
				<ul class="nobullet">
					<li>
						<h2>Education</h2>
						<ul class="nobullet">
							{CVData.education.map((data, index) => {
								return <ul class="nobullet">
									<li><b>{data.university}</b></li>
									<li><i>{data.degree}</i></li>
									<li>{data.subject}</li>
									<li>{data.dates}</li>
									<br/>
								</ul>
							})}
						</ul>
					</li>
					<li>
						<h2>Research Experience</h2>
						<ul class="nobullet">
							{CVData.research.map((data, idx) => {
								return <ul class="nobullet">
									<li><b>{data.group}</b></li>
									<li><i>{data.role}</i></li>
									<li>{data.dates}</li>
									<ul>
										{CVData.research[idx].duties.map((duty, idx_) => {
											return <li>{duty}</li>
										})}
									</ul>
									<br/>
								</ul>
							})}
						</ul>
					</li>
					<li>
						<h2>Teaching Experience</h2>
						<ul class="nobullet">
							{CVData.teaching.map((data, idx) => {
								return <ul class="nobullet">
									<li><b>{data.university}: {data.course}</b></li>
									<li><i>{data.role}</i></li>
									<li>{data.dates}</li>
									<ul>
										{CVData.teaching[idx].duties.map((duty, idx_) => {
											return <li>{duty}</li>
										})}
									</ul>
									<br/>
								</ul>
							})}
						</ul>
					</li>
					<li>
						<h2>Professional Experience</h2>
						<ul class="nobullet">
							{CVData.professional.map((data, idx) => {
								return <ul class="nobullet">
									<li><b>{data.employer}</b></li>
									<li><i>{data.role}</i></li>
									<li>{data.dates}</li>
									<ul>
										{CVData.professional[idx].duties.map((duty, idx_) => {
											return <li>{duty}</li>
										})}
									</ul>
									<br/>
								</ul>
							})}
						</ul>
					</li>
				</ul>
			</div>
		</Layout>
		<Sidebar/>
	</div>
)
